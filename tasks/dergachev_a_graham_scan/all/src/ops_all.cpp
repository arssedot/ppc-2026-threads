#include "dergachev_a_graham_scan/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "dergachev_a_graham_scan/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dergachev_a_graham_scan {

namespace {

using Pt = std::pair<double, double>;

double CrossProduct(const Pt &o, const Pt &a, const Pt &b) {
  return ((a.first - o.first) * (b.second - o.second)) - ((a.second - o.second) * (b.first - o.first));
}

double DistSquared(const Pt &a, const Pt &b) {
  double dx = a.first - b.first;
  double dy = a.second - b.second;
  return (dx * dx) + (dy * dy);
}

const double kPi = std::acos(-1.0);

bool IsLowerLeft(const Pt &a, const Pt &b) {
  return a.second < b.second || (a.second == b.second && a.first < b.first);
}

int FindPivotIndex(const std::vector<Pt> &pts, int num_threads) {
  int n = static_cast<int>(pts.size());
  if (n <= 1) {
    return 0;
  }
  if (n < num_threads * 2) {
    int pivot_idx = 0;
    for (int i = 1; i < n; i++) {
      if (IsLowerLeft(pts[i], pts[pivot_idx])) {
        pivot_idx = i;
      }
    }
    return pivot_idx;
  }

  std::vector<int> local_best(num_threads);
#pragma omp parallel num_threads(num_threads) default(none) shared(pts, n, local_best, num_threads)
  {
    int tid = omp_get_thread_num();
    int lo = (tid * n) / num_threads;
    int hi = ((tid + 1) * n) / num_threads;
    int best = lo;
    for (int i = lo + 1; i < hi; i++) {
      if (IsLowerLeft(pts[i], pts[best])) {
        best = i;
      }
    }
    local_best[tid] = best;
  }

  int pivot_idx = local_best[0];
  for (int t = 1; t < num_threads; t++) {
    if (IsLowerLeft(pts[local_best[t]], pts[pivot_idx])) {
      pivot_idx = local_best[t];
    }
  }
  return pivot_idx;
}

void ParallelSortByAngle(std::vector<Pt> &pts, const Pt &pivot, int num_threads) {
  int n = static_cast<int>(pts.size());
  int sort_count = n - 1;

  auto cmp = [&pivot](const Pt &a, const Pt &b) {
    double cross = CrossProduct(pivot, a, b);
    if (cross > 0.0) {
      return true;
    }
    if (cross < 0.0) {
      return false;
    }
    return DistSquared(pivot, a) < DistSquared(pivot, b);
  };

  if (num_threads <= 1 || sort_count <= num_threads) {
    std::sort(pts.begin() + 1, pts.end(), cmp);
    return;
  }

  const int chunk = sort_count / num_threads;
#pragma omp parallel num_threads(num_threads) default(none) shared(pts, n, chunk, cmp, num_threads)
  {
    const int tid = omp_get_thread_num();
    const int lo = 1 + (tid * chunk);
    const int hi = (tid == num_threads - 1) ? n : 1 + ((tid + 1) * chunk);
    std::sort(pts.begin() + lo, pts.begin() + hi, cmp);
  }

  int boundary = 1 + chunk;
  for (int tid = 1; tid < num_threads; tid++) {
    const int next = (tid == num_threads - 1) ? n : 1 + ((tid + 1) * chunk);
    std::inplace_merge(pts.begin() + 1, pts.begin() + boundary, pts.begin() + next, cmp);
    boundary = next;
  }
}

void BuildHull(std::vector<Pt> &pts, std::vector<Pt> &hull) {
  hull.clear();
  int n = static_cast<int>(pts.size());
  if (n <= 1) {
    if (!pts.empty()) {
      hull.push_back(pts[0]);
    }
    return;
  }
  if (std::all_of(pts.begin() + 1, pts.end(),
                  [&](const Pt &p) { return p.first == pts[0].first && p.second == pts[0].second; })) {
    hull.push_back(pts[0]);
    return;
  }

  const int num_threads = ppc::util::GetNumThreads();
  int pivot = FindPivotIndex(pts, num_threads);
  std::swap(pts[0], pts[pivot]);
  ParallelSortByAngle(pts, pts[0], num_threads);

  for (const auto &p : pts) {
    while (hull.size() > 1 && CrossProduct(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
}

void FlattenInto(const std::vector<Pt> &pts, std::vector<double> &flat) {
  const size_t count = pts.size();
  flat.resize(count * 2);
#pragma omp parallel for default(none) shared(pts, flat, count) num_threads(ppc::util::GetNumThreads())
  for (int i = 0; i < static_cast<int>(count); i++) {
    flat[static_cast<size_t>(i) * 2] = pts[i].first;
    flat[(static_cast<size_t>(i) * 2) + 1] = pts[i].second;
  }
}

void UnflattenInto(const std::vector<double> &flat, std::vector<Pt> &pts) {
  pts.resize(flat.size() / 2);
  const int n = static_cast<int>(pts.size());
#pragma omp parallel for default(none) shared(flat, pts, n) num_threads(ppc::util::GetNumThreads())
  for (int i = 0; i < n; i++) {
    pts[static_cast<size_t>(i)] = {flat[static_cast<size_t>(i) * 2], flat[(static_cast<size_t>(i) * 2) + 1]};
  }
}

int TotalPointCount(int n_input) {
  if (n_input <= 0) {
    return 0;
  }
  return n_input + (n_input > 3 ? 1 : 0);
}

struct WorkBuffers {
  std::vector<double> flat_all;
  std::vector<double> local_flat;
  std::vector<double> hull_flat;
  std::vector<double> gathered_flat;
  std::vector<double> result_flat;
  std::vector<Pt> local_pts;
  std::vector<Pt> merged_pts;
};

WorkBuffers &GetWorkBuffers() {
  static thread_local WorkBuffers buffers;
  return buffers;
}

}  // namespace

DergachevAGrahamScanALL::DergachevAGrahamScanALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DergachevAGrahamScanALL::ValidationImpl() {
  return GetInput() >= 0;
}

bool DergachevAGrahamScanALL::PreProcessingImpl() {
  hull_.clear();
  int n_input = GetInput();
  if (n_input <= 0) {
    points_.clear();
    return true;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    points_.clear();
    return true;
  }

  points_.resize(n_input);
  double step = (2.0 * kPi) / n_input;
  auto *pts_data = points_.data();
#pragma omp parallel for default(none) shared(pts_data, step, n_input) num_threads(ppc::util::GetNumThreads())
  for (int i = 0; i < n_input; i++) {
    pts_data[i] = {std::cos(step * i), std::sin(step * i)};
  }
  if (n_input > 3) {
    points_.emplace_back(0.0, 0.0);
  }
  return true;
}

bool DergachevAGrahamScanALL::RunImpl() {
  hull_.clear();

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int n_input = GetInput();
  int n = TotalPointCount(n_input);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank == 0 && !points_.empty()) {
      hull_.push_back(points_[0]);
    }
    int hull_size = static_cast<int>(hull_.size());
    MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0 && hull_size == 1) {
      hull_.assign(1, Pt{});
    }
    if (hull_size == 1) {
      std::vector<double> pt_flat(2);
      if (rank == 0) {
        pt_flat[0] = hull_[0].first;
        pt_flat[1] = hull_[0].second;
      }
      MPI_Bcast(pt_flat.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank != 0) {
        hull_[0] = {pt_flat[0], pt_flat[1]};
      }
    }
    return true;
  }

  int all_same = 0;
  if (rank == 0) {
    all_same = std::all_of(points_.begin() + 1, points_.end(),
                           [&](const Pt &p) { return p.first == points_[0].first && p.second == points_[0].second; })
                   ? 1
                   : 0;
  }
  MPI_Bcast(&all_same, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (all_same != 0) {
    if (rank == 0) {
      hull_.push_back(points_[0]);
    }
    int hull_size = 1;
    MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    hull_.resize(static_cast<size_t>(hull_size));
    auto &buffers = GetWorkBuffers();
    buffers.result_flat.resize(2);
    if (rank == 0) {
      buffers.result_flat[0] = hull_[0].first;
      buffers.result_flat[1] = hull_[0].second;
    }
    MPI_Bcast(buffers.result_flat.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      hull_[0] = {buffers.result_flat[0], buffers.result_flat[1]};
    }
    return true;
  }

  if (world_size == 1) {
    BuildHull(points_, hull_);
    return true;
  }

  auto &buffers = GetWorkBuffers();
  std::vector<int> send_counts(world_size);
  std::vector<int> send_displs(world_size);
  int disp = 0;
  for (int i = 0; i < world_size; i++) {
    int chunk = (n / world_size) + ((i < (n % world_size)) ? 1 : 0);
    send_counts[i] = chunk * 2;
    send_displs[i] = disp;
    disp += send_counts[i];
  }

  if (rank == 0) {
    FlattenInto(points_, buffers.flat_all);
  }

  const int local_size = send_counts[rank];
  buffers.local_flat.resize(static_cast<size_t>(local_size));
  MPI_Scatterv(rank == 0 ? buffers.flat_all.data() : nullptr, send_counts.data(), send_displs.data(), MPI_DOUBLE,
               buffers.local_flat.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  UnflattenInto(buffers.local_flat, buffers.local_pts);
  std::vector<Pt> local_hull;
  BuildHull(buffers.local_pts, local_hull);

  const int local_hull_flat_size = static_cast<int>(local_hull.size()) * 2;
  std::vector<int> recv_counts(world_size);
  MPI_Gather(&local_hull_flat_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> recv_displs(world_size);
  int total_recv = 0;
  if (rank == 0) {
    for (int i = 0; i < world_size; i++) {
      recv_displs[i] = total_recv;
      total_recv += recv_counts[i];
    }
  }

  FlattenInto(local_hull, buffers.hull_flat);
  buffers.gathered_flat.resize(static_cast<size_t>(total_recv));
  MPI_Gatherv(buffers.hull_flat.data(), local_hull_flat_size, MPI_DOUBLE, buffers.gathered_flat.data(),
              recv_counts.data(), recv_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    UnflattenInto(buffers.gathered_flat, buffers.merged_pts);
    BuildHull(buffers.merged_pts, hull_);
  }

  int hull_size = static_cast<int>(hull_.size());
  MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  hull_.resize(static_cast<size_t>(hull_size));
  buffers.result_flat.resize(static_cast<size_t>(hull_size) * 2);
  if (rank == 0) {
    FlattenInto(hull_, buffers.result_flat);
  }
  MPI_Bcast(buffers.result_flat.data(), hull_size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    UnflattenInto(buffers.result_flat, hull_);
  }

  return true;
}

bool DergachevAGrahamScanALL::PostProcessingImpl() {
  GetOutput() = static_cast<int>(hull_.size());
  return true;
}

}  // namespace dergachev_a_graham_scan
