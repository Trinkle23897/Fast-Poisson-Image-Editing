#include <pybind11/pybind11.h>

class PIE {
 public:
  PIE(int n) { printf("2333\n"); }
  int run() { return 6; }
};

namespace py = pybind11;

PYBIND11_MODULE(pie_core, m) {
  py::class_<PIE>(m, "PIE").def(py::init<int>()).def("run", &PIE::run);
}
