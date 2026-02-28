#include "dergachev_a_graham_scan_all/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
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
    flat[2 * i] = pts[i].x;
    flat[2 * i + 1] = pts[i].y;
  }
  return flat;
}

std::vector<Point> Unflatten(const std::vector<double> &flat) {
  int count = static_cast<int>(flat.size()) / 2;
  std::vector<Point> pts(count);
  for (int i = 0; i < count; i++) {
    pts[i].x = flat[2 * i];
    pts[i].y = flat[2 * i + 1];
  }
  return pts;
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
  int chunk = n / num_threads;
  int remainder = n % num_threads;
  int offset = 0;
  for (int t = 0; t < num_threads; t++) {
    int sz = chunk + (t < remainder ? 1 : 0);
    threads.emplace_back([&thread_hulls, &pts, offset, sz, t]() {
      std::vector<Point> local(pts.begin() + offset, pts.begin() + offset + sz);
      thread_hulls[t] = GrahamScan(std::move(local));
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
    hull_ = points_;
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

  int base = n / world_size;
  int rem = n % world_size;
  std::vector<Point> local_chunk;

  if (rank == 0) {
    int my_sz = base + (0 < rem ? 1 : 0);
    local_chunk.assign(points_.begin(), points_.begin() + my_sz);
    int off = my_sz;
    for (int i = 1; i < world_size; i++) {
      int sz = base + (i < rem ? 1 : 0);
      MPI_Send(&sz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      if (sz > 0) {
        std::vector<double> buf = Flatten({points_.begin() + off, points_.begin() + off + sz});
        MPI_Send(buf.data(), sz * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      }
      off += sz;
    }
  } else {
    int sz = 0;
    MPI_Recv(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (sz > 0) {
      std::vector<double> buf(sz * 2);
      MPI_Recv(buf.data(), sz * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_chunk = Unflatten(buf);
    }
  }

  std::vector<Point> local_hull;
  if (!local_chunk.empty()) {
    local_hull = GrahamScanThreaded(local_chunk);
  }

  if (rank == 0) {
    std::vector<Point> all_hull(local_hull.begin(), local_hull.end());
    for (int i = 1; i < world_size; i++) {
      int hsz = 0;
      MPI_Recv(&hsz, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (hsz > 0) {
        std::vector<double> buf(hsz * 2);
        MPI_Recv(buf.data(), hsz * 2, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<Point> rh = Unflatten(buf);
        all_hull.insert(all_hull.end(), rh.begin(), rh.end());
      }
    }
    hull_ = GrahamScan(std::move(all_hull));

    int final_sz = static_cast<int>(hull_.size());
    std::vector<double> flat_hull = Flatten(hull_);
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&final_sz, 1, MPI_INT, i, 4, MPI_COMM_WORLD);
      if (final_sz > 0) {
        MPI_Send(flat_hull.data(), final_sz * 2, MPI_DOUBLE, i, 5, MPI_COMM_WORLD);
      }
    }
  } else {
    int hsz = static_cast<int>(local_hull.size());
    MPI_Send(&hsz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    if (hsz > 0) {
      std::vector<double> buf = Flatten(local_hull);
      MPI_Send(buf.data(), hsz * 2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    int final_sz = 0;
    MPI_Recv(&final_sz, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (final_sz > 0) {
      std::vector<double> buf(final_sz * 2);
      MPI_Recv(buf.data(), final_sz * 2, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      hull_ = Unflatten(buf);
    }
  }

  return true;
}

bool DergachevAGrahamScanALL::PostProcessingImpl() {
  GetOutput() = static_cast<int>(hull_.size());
  return true;
}

}  // namespace dergachev_a_graham_scan_all
