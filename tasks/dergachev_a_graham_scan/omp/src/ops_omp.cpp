#include "dergachev_a_graham_scan/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
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

bool AngleLess(const Pt &a, const Pt &b, const Pt &pivot) {
  const double cross = CrossProduct(pivot, a, b);
  return cross > 0.0 || (cross == 0.0 && DistSquared(pivot, a) < DistSquared(pivot, b));
}

int FindPivotIndex(const std::vector<Pt> &pts) {
  int best = 0;
  for (int i = 1; std::cmp_less(i, pts.size()); i++) {
    if (IsLowerLeft(pts[i], pts[best])) {
      best = i;
    }
  }
  return best;
}

void SortByAngle(std::vector<Pt> &pts, const Pt &pivot) {
  const int sort_count = static_cast<int>(pts.size()) - 1;
  const int num_threads = ppc::util::GetNumThreads();
  const int n = static_cast<int>(pts.size());
  if (sort_count <= 1) {
    return;
  }

  auto cmp = [&](const Pt &a, const Pt &b) { return AngleLess(a, b, pivot); };

  if (num_threads <= 1) {
    std::sort(pts.begin() + 1, pts.end(), cmp);
    return;
  }

  if (num_threads == 2) {
    const int mid = 1 + (sort_count / 2);
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
      {
        std::sort(pts.begin() + 1, pts.begin() + mid, cmp);
      }
#pragma omp section
      {
        std::sort(pts.begin() + mid, pts.end(), cmp);
      }
    }
    std::inplace_merge(pts.begin() + 1, pts.begin() + mid, pts.end(), cmp);
    return;
  }

  const int chunk = sort_count / num_threads;
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int tid = 0; tid < num_threads; tid++) {
    const int lo = 1 + (tid * chunk);
    const int hi = (tid == num_threads - 1) ? n : 1 + ((tid + 1) * chunk);
    std::sort(pts.begin() + lo, pts.begin() + hi, cmp);
  }

  int boundary = 1 + chunk;
  for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
    const int next = (thread_idx == num_threads - 1) ? n : 1 + ((thread_idx + 1) * chunk);
    std::inplace_merge(pts.begin() + 1, pts.begin() + boundary, pts.begin() + next, cmp);
    boundary = next;
  }
}

}  // namespace

DergachevAGrahamScanOMP::DergachevAGrahamScanOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DergachevAGrahamScanOMP::ValidationImpl() {
  return GetInput() >= 0;
}

bool DergachevAGrahamScanOMP::PreProcessingImpl() {
  hull_.clear();
  int n = GetInput();
  if (n <= 0) {
    points_.clear();
    return true;
  }
  points_.resize(n);
  double step = (2.0 * kPi) / n;
  auto *pts_data = points_.data();
#pragma omp parallel for schedule(static) default(none) shared(pts_data, step, n) \
    num_threads(ppc::util::GetNumThreads())
  for (int i = 0; i < n; i++) {
    pts_data[i] = {std::cos(step * i), std::sin(step * i)};
  }
  if (n > 3) {
    points_.emplace_back(0.0, 0.0);
  }
  return true;
}

bool DergachevAGrahamScanOMP::RunImpl() {
  hull_.clear();

  thread_local std::vector<Pt> pts;
  pts.assign(points_.begin(), points_.end());
  const int n = static_cast<int>(pts.size());

  if (n <= 1 || std::all_of(pts.begin() + 1, pts.end(),
                            [&](const Pt &pt) { return pt.first == pts[0].first && pt.second == pts[0].second; })) {
    if (!pts.empty()) {
      hull_.push_back(pts[0]);
    }
    return true;
  }

  const int pivot_idx = FindPivotIndex(pts);
  std::swap(pts[0], pts[static_cast<size_t>(pivot_idx)]);

  const Pt pivot = pts[0];
  SortByAngle(pts, pivot);

  hull_.reserve(static_cast<size_t>(n));
  for (const auto &p : pts) {
    while (hull_.size() > 1 && CrossProduct(hull_[hull_.size() - 2], hull_.back(), p) <= 0.0) {
      hull_.pop_back();
    }
    hull_.push_back(p);
  }

  return true;
}

bool DergachevAGrahamScanOMP::PostProcessingImpl() {
  GetOutput() = static_cast<int>(hull_.size());
  return true;
}

}  // namespace dergachev_a_graham_scan
