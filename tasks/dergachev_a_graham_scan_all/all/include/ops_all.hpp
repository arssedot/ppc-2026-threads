#pragma once

#include <vector>

#include "dergachev_a_graham_scan_all/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergachev_a_graham_scan_all {

struct Point {
  double x;
  double y;
};

class DergachevAGrahamScanALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit DergachevAGrahamScanALL(const InType &in);

  void SetPoints(const std::vector<Point> &pts);
  [[nodiscard]] std::vector<Point> GetHull() const;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> points_;
  std::vector<Point> hull_;
  bool custom_points_ = false;
};

}  // namespace dergachev_a_graham_scan_all
