#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "skvq_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("skvq_quant_fake", &skvq_quant_fake);
  m.def("skvq_quant_pack", &skvq_quant_pack);
  m.def("skvq_dequant_unpack", &skvq_dequant_unpack);
}