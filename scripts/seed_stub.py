"""Build a minimal CUDA, Python, or Triton stub from a task definition."""
from __future__ import annotations

import re


ADAPTERS = ("flashinfer", "sol")
STUB_LANGUAGES = ("cuda", "python", "triton")
CUDA_BINDINGS = ("tvm-ffi", "torch")
ADAPTER_LANGUAGES = {
    "flashinfer": {
        "cuda": "cuda",
        "python": "python",
        "triton": "triton",
    },
    "sol": {
        "cuda": "cuda_cpp",
        "python": "pytorch",
        "triton": "triton",
    },
}
CPP_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

INTEGER_DTYPES = {
    "byte",
    "char",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "long",
    "short",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
FLOAT_DTYPES = {
    "bfloat16",
    "double",
    "float",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float16",
    "float32",
    "float64",
    "half",
}


def _validate_definition(definition: dict) -> None:
    if not isinstance(definition, dict):
        raise SystemExit("Error: the task definition must be a JSON object.")
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        raise SystemExit("Error: the task definition has no name.")
    for collection in ("inputs", "outputs"):
        fields = definition.get(collection)
        if not isinstance(fields, dict) or not fields:
            raise SystemExit(
                f"Error: the task must define at least one {collection[:-1]}."
            )
        for field_name, field in fields.items():
            if not isinstance(field_name, str) or not CPP_IDENTIFIER.fullmatch(
                field_name
            ):
                raise SystemExit(
                    f"Error: unsupported {collection[:-1]} name "
                    f"{field_name!r}; C++/Python identifiers are required."
                )
            if not isinstance(field, dict):
                raise SystemExit(
                    f"Error: field {field_name!r} must be an object."
                )
            dtype = field.get("dtype")
            if not isinstance(dtype, str) or not dtype:
                raise SystemExit(
                    f"Error: field {field_name!r} has no dtype."
                )


def _normalized_dtype(dtype: str) -> str:
    return dtype.lower().rsplit(".", 1)[-1]


def _scalar_cpp_type(field_name: str, field: dict) -> str:
    dtype = _normalized_dtype(field["dtype"])
    if dtype == "bool":
        return "bool"
    if dtype in INTEGER_DTYPES:
        # PyBind receives Python integers; the kernel can narrow explicitly.
        return "int64_t"
    if dtype in FLOAT_DTYPES:
        # PyBind receives Python floats; the kernel can narrow explicitly.
        return "double"
    raise SystemExit(
        f"Error: scalar field {field_name!r} has unsupported dtype "
        f"{field['dtype']!r}."
    )


def _field_cpp_type(field_name: str, field: dict) -> str:
    if field.get("shape") is not None:
        return "torch::Tensor"
    return _scalar_cpp_type(field_name, field)


def _tvm_ffi_field_cpp_type(field_name: str, field: dict) -> str:
    if field.get("shape") is not None:
        return "tvm::ffi::TensorView"
    return _scalar_cpp_type(field_name, field)


def _arguments(definition: dict) -> list[tuple[str, str, str]]:
    return [
        (name, f"arg_{name}", _field_cpp_type(name, field))
        for name, field in definition["inputs"].items()
    ]


def _return_type(definition: dict) -> str:
    types = [
        _field_cpp_type(name, field)
        for name, field in definition["outputs"].items()
    ]
    if len(types) == 1:
        return types[0]
    return f"std::tuple<{', '.join(types)}>"


def _function_declaration(
    return_type: str,
    arguments: list[tuple[str, str, str]],
    *,
    name: str = "run_cuda",
    terminator: str,
) -> str:
    if not arguments:
        return f"{return_type} {name}(){terminator}"
    rendered = ",\n".join(
        f"    {cpp_type} {cpp_name}"
        for _, cpp_name, cpp_type in arguments
    )
    return f"{return_type} {name}(\n{rendered}\n){terminator}"


def _render_torch_cuda_sources(definition: dict) -> list[dict[str, str]]:
    arguments = _arguments(definition)
    return_type = _return_type(definition)
    definition_start = _function_declaration(
        return_type, arguments, name="run", terminator=" {"
    )

    header = "#pragma once\n#include <torch/extension.h>\n"
    kernel = '#include "kernel.h"\n'

    pybind_args = "\n".join(
        f'        py::arg("{name}"){"," if index < len(arguments) - 1 else ""}'
        for index, (name, _, _) in enumerate(arguments)
    )
    binding_tail = f",\n{pybind_args}" if pybind_args else ""
    main_cpp = (
        "#include <cstdint>\n"
        "#include <tuple>\n"
        "#include <pybind11/stl.h>\n"
        '#include "kernel.h"\n\n'
        f"{definition_start}\n"
        '    TORCH_CHECK(false, "Not implemented");\n'
        "    return {};\n"
        "}\n\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '    m.def("run", &run'
        f"{binding_tail});\n"
        "}\n"
    )

    return [
        {"path": "kernel.cu", "content": kernel},
        {"path": "kernel.h", "content": header},
        {"path": "main.cpp", "content": main_cpp},
    ]


def _render_tvm_ffi_cuda_sources(definition: dict) -> list[dict[str, str]]:
    arguments = [
        (name, f"arg_{name}", _tvm_ffi_field_cpp_type(name, field))
        for collection in ("inputs", "outputs")
        for name, field in definition[collection].items()
    ]
    declaration = _function_declaration(
        "void", arguments, name="run_cuda", terminator=";"
    )
    definition_start = _function_declaration(
        "void", arguments, name="run_cuda", terminator=" {"
    )
    tensor_inputs = [
        f"arg_{name}"
        for name, field in definition["inputs"].items()
        if field.get("shape") is not None
    ]
    tensor_outputs = [
        f"arg_{name}"
        for name, field in definition["outputs"].items()
        if field.get("shape") is not None
    ]
    example_tensor = (tensor_inputs or tensor_outputs)[0]
    example_output = tensor_outputs[0]

    header = (
        "#pragma once\n"
        "#include <cstdint>\n"
        "#include <tvm/ffi/container/tensor.h>\n\n"
        f"{declaration}\n"
    )
    kernel = (
        '#include "kernel.h"\n'
        "#include <cuda_runtime.h>\n"
        "#include <tvm/ffi/error.h>\n"
        "#include <tvm/ffi/extra/c_env_api.h>\n\n"
        f"{definition_start}\n"
        "    // TensorView is non-owning. Read shapes and cast data_ptr() to the\n"
        "    // element type required by the CUDA kernel. Outputs are supplied by\n"
        "    // the caller and must be written in place.\n"
        f"    const int64_t example_numel = {example_tensor}.numel();\n"
        f"    const int64_t example_first_dim = {example_tensor}.ndim() == 0\n"
        f"        ? 1 : {example_tensor}.shape()[0];\n"
        f"    const void* example_input_data = {example_tensor}.data_ptr();\n"
        f"    void* example_output_data = {example_output}.data_ptr();\n\n"
        "    // Always launch on TVM-FFI's current stream, not CUDA's default stream.\n"
        f"    const DLDevice device = {example_tensor}.device();\n"
        "    cudaStream_t stream = static_cast<cudaStream_t>(\n"
        "        TVMFFIEnvGetStream(device.device_type, device.device_id));\n\n"
        "    (void)example_numel;\n"
        "    (void)example_first_dim;\n"
        "    (void)example_input_data;\n"
        "    (void)example_output_data;\n"
        "    (void)stream;\n"
        '    TVM_FFI_THROW(RuntimeError) << "Not implemented";\n'
        "}\n"
    )
    main_cpp = (
        "#include <tvm/ffi/function.h>\n"
        '#include "kernel.h"\n\n'
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_cuda);\n"
    )

    return [
        {"path": "kernel.cu", "content": kernel},
        {"path": "kernel.h", "content": header},
        {"path": "main.cpp", "content": main_cpp},
    ]


def _render_python_source(definition: dict, language: str) -> list[dict[str, str]]:
    names = list(definition["inputs"])
    if names:
        parameters = ",\n".join(f"    {name}" for name in names)
        signature = f"def run(\n{parameters},\n):"
    else:
        signature = "def run():"

    imports = "import torch\n"
    guidance = ""
    if language == "triton":
        imports += "import triton\nimport triton.language as tl\n"
        guidance = "\n# Define one or more @triton.jit kernels above run().\n"

    content = (
        f"{imports}"
        f"{guidance}\n\n"
        f"{signature}\n"
        '    raise NotImplementedError("Kernel not implemented")\n'
    )
    return [{"path": "main.py", "content": content}]


def _render_sources(
    definition: dict,
    language: str,
    cuda_binding: str,
) -> list[dict[str, str]]:
    if language == "cuda":
        if cuda_binding == "tvm-ffi":
            return _render_tvm_ffi_cuda_sources(definition)
        return _render_torch_cuda_sources(definition)
    return _render_python_source(definition, language)


def _build_spec(
    adapter: str,
    hardware: str,
    language: str,
    cuda_binding: str,
) -> dict:
    uses_tvm_ffi = (
        adapter == "flashinfer"
        and language == "cuda"
        and cuda_binding == "tvm-ffi"
    )
    common = {
        "target_hardware": [hardware],
        "entry_point": "main.cpp::run" if language == "cuda" else "main.py::run",
        "dependencies": [],
        "destination_passing_style": uses_tvm_ffi,
        "binding": "tvm-ffi" if uses_tvm_ffi else "torch",
    }
    backend_language = ADAPTER_LANGUAGES[adapter][language]
    if adapter == "flashinfer":
        return {"language": backend_language, **common}
    if hardware not in ("LOCAL", "B200"):
        raise SystemExit(
            "Error: SOL --hardware must be LOCAL or B200."
        )
    return {"languages": [backend_language], **common}


def make_stub_solution(
    definition: dict,
    adapter: str,
    hardware: str,
    language: str,
    cuda_binding: str | None = None,
) -> dict:
    _validate_definition(definition)
    if adapter not in ADAPTERS:
        raise SystemExit(f"Error: unknown stub adapter {adapter!r}.")
    if language not in STUB_LANGUAGES:
        available = ", ".join(STUB_LANGUAGES)
        raise SystemExit(
            f"Error: unknown stub language {language!r}; choose {available}."
        )
    if cuda_binding is not None and cuda_binding not in CUDA_BINDINGS:
        available = ", ".join(CUDA_BINDINGS)
        raise SystemExit(
            f"Error: unknown CUDA binding {cuda_binding!r}; choose {available}."
        )
    if cuda_binding is not None and (adapter != "flashinfer" or language != "cuda"):
        raise SystemExit(
            "Error: a CUDA binding can only be selected for a FlashInfer CUDA stub."
        )
    resolved_cuda_binding = (
        cuda_binding
        if cuda_binding is not None
        else "tvm-ffi" if adapter == "flashinfer" and language == "cuda" else "torch"
    )
    return {
        "name": f"{definition['name']}_{language}_stub",
        "definition": definition["name"],
        "author": "agent",
        "spec": _build_spec(adapter, hardware, language, resolved_cuda_binding),
        "sources": _render_sources(definition, language, resolved_cuda_binding),
        "description": f"Generated {language} stub.",
    }
