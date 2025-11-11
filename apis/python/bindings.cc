#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "infeng/runtime/runtime.h"
#include "infeng/tokenizer/tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(pyinfeng, m) {
  m.doc() = "Python bindings for INF-ENG CPU inference engine (prototype)";

  py::class_<infeng::runtime::EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("num_decode_threads", &infeng::runtime::EngineConfig::num_decode_threads)
      .def_readwrite("num_adapter_threads", &infeng::runtime::EngineConfig::num_adapter_threads)
      .def_readwrite("num_tokenizer_threads", &infeng::runtime::EngineConfig::num_tokenizer_threads)
      .def_readwrite("enable_metrics", &infeng::runtime::EngineConfig::enable_metrics);

  m.def("initialize", &infeng::runtime::initialize, "Initialize the INF-ENG runtime",
        py::arg("config"));
  m.def("shutdown", &infeng::runtime::shutdown, "Shutdown the INF-ENG runtime");
  m.def("version", &infeng::runtime::version, "Return the INF-ENG engine version");

  py::class_<infeng::tokenizer::Tokenizer>(m, "Tokenizer")
      .def(py::init<const std::string&>(), py::arg("model_path"))
      .def("enable_prefix_cache", &infeng::tokenizer::Tokenizer::enable_prefix_cache,
           py::arg("enable"))
      .def("set_prefix_params", &infeng::tokenizer::Tokenizer::set_prefix_params,
           py::arg("prefix_k"), py::arg("capacity"))
      .def("set_thread_override", &infeng::tokenizer::Tokenizer::set_thread_override,
           py::arg("threads"))
      .def("encode_stream_begin",
           [](infeng::tokenizer::Tokenizer& self, const std::string& text, std::size_t threads) {
             self.encode_stream_begin(text, threads);
           },
           py::arg("text"), py::arg("threads") = 0)
      .def("encode_stream_next",
           [](infeng::tokenizer::Tokenizer& self) -> py::object {
             infeng::tokenizer::Batch batch{};
             if (!self.encode_stream_next(&batch)) {
               return py::none();
             }
             auto array = py::array_t<std::int32_t>(static_cast<py::ssize_t>(batch.len));
             auto buf = array.mutable_unchecked<1>();
             for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
               buf(i) = batch.ids[i];
             }
             return array;
           })
      .def("encode_stream_end", &infeng::tokenizer::Tokenizer::encode_stream_end);
}
