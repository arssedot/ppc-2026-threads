#include "dergachev_a_graham_scan/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "dergachev_a_graham_scan/common/include/common.hpp"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/parallel_sort.h"

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

int FindPivotIndex(const std::vector<Pt> &pts, int n) {
  return tbb::parallel_reduce(tbb::blocked_range<int>(1, n), 0,
                              [&](const tbb::blocked_range<int> &range, int best) -> int {
    for (int i = range.begin(); i < range.end(); ++i) {
      if (IsLowerLeft(pts[static_cast<std::size_t>(i)], pts[static_cast<std::size_t>(best)])) {
        best = i;
      }
    }
    return best;
  }, [&](int a, int b) -> int {
    return IsLowerLeft(pts[static_cast<std::size_t>(a)], pts[static_cast<std::size_t>(b)]) ? a : b;
  });
}

void SortByAngle(std::vector<Pt> &pts) {
  Pt pivot = pts[0];
  tbb::parallel_sort(pts.begin() + 1, pts.end(), [&pivot](const Pt &a, const Pt &b) {
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

}  // namespace

DergachevAGrahamScanTBB::DergachevAGrahamScanTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DergachevAGrahamScanTBB::ValidationImpl() {
  return GetInput() >= 0;
}

bool DergachevAGrahamScanTBB::PreProcessingImpl() {
  hull_.clear();
  int n = GetInput();
  if (n <= 0) {
    points_.clear();
    return true;
  }
  points_.resize(n);
  double step = (2.0 * kPi) / n;
  tbb::parallel_for(0, n,
                    [&](int i) { points_[static_cast<std::size_t>(i)] = {std::cos(step * i), std::sin(step * i)}; });
  if (n > 3) {
    points_.emplace_back(0.0, 0.0);
  }
  return true;
}

bool DergachevAGrahamScanTBB::RunImpl() {
  hull_.clear();
  std::vector<Pt> pts(points_.begin(), points_.end());
  int n = static_cast<int>(pts.size());

  if (n <= 1 || std::all_of(pts.begin() + 1, pts.end(),
                            [&](const Pt &pt) { return pt.first == pts[0].first && pt.second == pts[0].second; })) {
    if (!pts.empty()) {
      hull_.push_back(pts[0]);
    }
    return true;
  }

  int pivot_idx = FindPivotIndex(pts, n);
  std::swap(pts[0], pts[static_cast<std::size_t>(pivot_idx)]);
  SortByAngle(pts);

  for (const auto &p : pts) {
    while (hull_.size() > 1 && CrossProduct(hull_[hull_.size() - 2], hull_.back(), p) <= 0.0) {
      hull_.pop_back();
    }
    hull_.push_back(p);
  }

  return true;
}

bool DergachevAGrahamScanTBB::PostProcessingImpl() {
  GetOutput() = static_cast<int>(hull_.size());
  return true;
}

}  // namespace dergachev_a_graham_scan
