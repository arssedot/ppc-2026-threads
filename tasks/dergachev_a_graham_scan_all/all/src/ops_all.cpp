#include "dergachev_a_graham_scan_all/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "dergachev_a_graham_scan_all/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_graham_scan_all {

namespace {

double CrossProduct(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

double DistSquared(const Point &a, const Point &b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return (dx * dx) + (dy * dy);
}

const double kPi = std::acos(-1.0);

bool AllPointsSame(const std::vector<Point> &pts) {
  for (int i = 1; std::cmp_less(i, pts.size()); i++) {
    if (pts[i].x != pts[0].x || pts[i].y != pts[0].y) {
      return false;
    }
  }
  return true;
}

int FindPivotIndex(const std::vector<Point> &pts) {
  int pivot_idx = 0;
  for (int i = 1; std::cmp_less(i, pts.size()); i++) {
    if (pts[i].y < pts[pivot_idx].y || (pts[i].y == pts[pivot_idx].y && pts[i].x < pts[pivot_idx].x)) {
      pivot_idx = i;
    }
  }
  return pivot_idx;
}

void SortByAngle(std::vector<Point> &pts) {
  Point pivot = pts[0];
  std::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    double cross = CrossProduct(pivot, a, b);
    if (cross > 0.0) {
      return true;
    }
    if (cross < 0.0) {
      return false;
    }
    return DistSquared(pivot, a) < DistSquared(pivot, b);
  });
}

std::vector<Point> GrahamScan(std::vector<Point> pts) {
  int n = static_cast<int>(pts.size());
  if (n <= 1) {
    return pts;
  }
  if (AllPointsSame(pts)) {
    return {pts[0]};
  }
  int pivot_idx = FindPivotIndex(pts);
  std::swap(pts[0], pts[pivot_idx]);
  SortByAngle(pts);
  std::vector<Point> hull;
  for (const auto &p : pts) {
    while (hull.size() > 1 && CrossProduct(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  return hull;
}

std::vector<double> Flatten(const std::vector<Point> &pts) {
  std::vector<double> flat(pts.size() * 2);
  for (int i = 0; std::cmp_less(i, pts.size()); i++) {
    auto idx = static_cast<size_t>(i) * 2;
    flat[idx] = pts[i].x;
    flat[idx + 1] = pts[i].y;
  }
  return flat;
}

std::vector<Point> Unflatten(const std::vector<double> &flat) {
  int count = static_cast<int>(flat.size()) / 2;
  std::vector<Point> pts(count);
  for (int i = 0; i < count; i++) {
    auto idx = static_cast<size_t>(i) * 2;
    pts[i].x = flat[idx];
    pts[i].y = flat[idx + 1];
  }
  return pts;
}

int ChunkSize(int idx, int total, int parts) {
  return (total / parts) + ((idx < (total % parts)) ? 1 : 0);
}

std::vector<Point> GrahamScanThreaded(const std::vector<Point> &pts) {
  int n = static_cast<int>(pts.size());
  if (n <= 1) {
    return pts;
  }
  if (AllPointsSame(pts)) {
    return {pts[0]};
  }
  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 1 || n < num_threads * 4) {
    return GrahamScan(pts);
  }
  std::vector<std::vector<Point>> thread_hulls(num_threads);
  std::vector<std::thread> threads;
  int offset = 0;
  for (int ti = 0; ti < num_threads; ti++) {
    int sz = ChunkSize(ti, n, num_threads);
    threads.emplace_back([&thread_hulls, &pts, offset, sz, ti]() {
      std::vector<Point> local(pts.begin() + offset, pts.begin() + offset + sz);
      thread_hulls[ti] = GrahamScan(std::move(local));
    });
    offset += sz;
  }
  for (auto &thr : threads) {
    thr.join();
  }
  std::vector<Point> merged;
  for (const auto &th : thread_hulls) {
    merged.insert(merged.end(), th.begin(), th.end());
  }
  return GrahamScan(std::move(merged));
}

void SendPoints(const std::vector<Point> &pts, int dest, int tag_sz, int tag_data) {
  int sz = static_cast<int>(pts.size());
  MPI_Send(&sz, 1, MPI_INT, dest, tag_sz, MPI_COMM_WORLD);
  if (sz > 0) {
    std::vector<double> buf = Flatten(pts);
    MPI_Send(buf.data(), sz * 2, MPI_DOUBLE, dest, tag_data, MPI_COMM_WORLD);
  }
}

std::vector<Point> RecvPoints(int source, int tag_sz, int tag_data) {
  int sz = 0;
  MPI_Recv(&sz, 1, MPI_INT, source, tag_sz, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (sz <= 0) {
    return {};
  }
  std::vector<double> buf(static_cast<size_t>(sz) * 2);
  MPI_Recv(buf.data(), sz * 2, MPI_DOUBLE, source, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return Unflatten(buf);
}

std::vector<Point> DistributePoints(int rank, int world_size, const std::vector<Point> &points) {
  int n = static_cast<int>(points.size());
  if (rank == 0) {
    int off = ChunkSize(0, n, world_size);
    for (int i = 1; i < world_size; i++) {
      int sz = ChunkSize(i, n, world_size);
      std::vector<Point> chunk(points.begin() + off, points.begin() + off + sz);
      SendPoints(chunk, i, 0, 1);
      off += sz;
    }
    return {points.begin(), points.begin() + ChunkSize(0, n, world_size)};
  }
  return RecvPoints(0, 0, 1);
}

std::vector<Point> GatherHulls(int rank, int world_size, const std::vector<Point> &local_hull) {
  if (rank == 0) {
    std::vector<Point> all_hull(local_hull);
    for (int i = 1; i < world_size; i++) {
      auto rh = RecvPoints(i, 2, 3);
      all_hull.insert(all_hull.end(), rh.begin(), rh.end());
    }
    auto result = GrahamScan(std::move(all_hull));
    for (int i = 1; i < world_size; i++) {
      SendPoints(result, i, 4, 5);
    }
    return result;
  }
  SendPoints(local_hull, 0, 2, 3);
  return RecvPoints(0, 4, 5);
}

}  // namespace

DergachevAGrahamScanALL::DergachevAGrahamScanALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

void DergachevAGrahamScanALL::SetPoints(const std::vector<Point> &pts) {
  points_.assign(pts.begin(), pts.end());
  custom_points_ = true;
}

std::vector<Point> DergachevAGrahamScanALL::GetHull() const {
  return hull_;
}

bool DergachevAGrahamScanALL::ValidationImpl() {
  return GetInput() >= 0;
}

bool DergachevAGrahamScanALL::PreProcessingImpl() {
  hull_.clear();
  if (!custom_points_) {
    int n = GetInput();
    if (n <= 0) {
      points_.clear();
      return true;
    }
    points_.resize(n);
    double step = (2.0 * kPi) / n;
    for (int i = 0; i < n; i++) {
      points_[i] = {.x = std::cos(step * i), .y = std::sin(step * i)};
    }
  }
  return true;
}

bool DergachevAGrahamScanALL::RunImpl() {
  hull_.clear();
  int n = static_cast<int>(points_.size());

  if (n <= 1) {
    hull_.assign(points_.begin(), points_.end());
    return true;
  }

  if (AllPointsSame(points_)) {
    hull_.push_back(points_[0]);
    return true;
  }

  int mpi_ready = 0;
  MPI_Initialized(&mpi_ready);

  int rank = 0;
  int world_size = 1;
  if (mpi_ready != 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  }

  if (world_size == 1) {
    hull_ = GrahamScanThreaded(points_);
    return true;
  }

  auto local_chunk = DistributePoints(rank, world_size, points_);
  std::vector<Point> local_hull;
  if (!local_chunk.empty()) {
    local_hull = GrahamScanThreaded(local_chunk);
  }
  hull_ = GatherHulls(rank, world_size, local_hull);
  return true;
}

bool DergachevAGrahamScanALL::PostProcessingImpl() {
  GetOutput() = static_cast<int>(hull_.size());
  return true;
}

}  // namespace dergachev_a_graham_scan_all
