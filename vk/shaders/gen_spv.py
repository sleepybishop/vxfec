#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VK_DIR = os.path.dirname(HERE)
OUT_HDR = os.path.join(VK_DIR, "vk_spv.h")
OUT_SRC = os.path.join(VK_DIR, "vk_spv.c")

SHADERS = [
    ("spv_ops_run_vec4_gf256", "ops_run.comp", ["-DUSE_VEC4=1"]),
    ("spv_ops_run_vec4_gf2", "ops_run.comp", ["-DUSE_VEC4=1", "-DPURE_GF2=1"]),
    ("spv_ops_run_vram_gf256", "ops_run.comp", []),
    ("spv_ops_run_vram_gf2", "ops_run.comp", ["-DPURE_GF2=1"]),
    ("spv_ops_mix_vec4", "ops_mix.comp", ["-DUSE_VEC4=1"]),
    ("spv_ops_mix", "ops_mix.comp", []),
    ("spv_ops_direct_gf2_vec4_16", "ops_direct_gf2.comp", ["-DUSE_VEC4=1", "-DTILE_E=16"]),
    ("spv_ops_direct_gf2_vec4_32", "ops_direct_gf2.comp", ["-DUSE_VEC4=1", "-DTILE_E=32"]),
    ("spv_ops_direct_gf2_16", "ops_direct_gf2.comp", ["-DTILE_E=16"]),
    ("spv_ops_direct_gf2_32", "ops_direct_gf2.comp", ["-DTILE_E=32"]),
    ("spv_ops_direct_gf256_nib_vec4_8", "ops_direct_gf256_nib.comp", ["-DUSE_VEC4=1", "-DTILE_E=8"]),
    ("spv_ops_direct_gf256_nib_vec4_16", "ops_direct_gf256_nib.comp", ["-DUSE_VEC4=1", "-DTILE_E=16"]),
    ("spv_ops_direct_gf256_nib_vec4_32", "ops_direct_gf256_nib.comp", ["-DUSE_VEC4=1", "-DTILE_E=32"]),
    ("spv_ops_direct_gf256_nib_8", "ops_direct_gf256_nib.comp", ["-DTILE_E=8"]),
    ("spv_ops_direct_gf256_nib_16", "ops_direct_gf256_nib.comp", ["-DTILE_E=16"]),
    ("spv_ops_direct_gf256_nib_32", "ops_direct_gf256_nib.comp", ["-DTILE_E=32"]),
    ("spv_ops_wire_gf2_vec4_16", "ops_wire_gf2_vec4.comp", ["-DTILE_E=16"]),
    ("spv_ops_wire_gf2_vec4_32", "ops_wire_gf2_vec4.comp", ["-DTILE_E=32"]),
    ("spv_ops_wire_gf256_nib_vec4_8", "ops_wire_gf256_nib_vec4.comp", ["-DTILE_E=8"]),
    ("spv_ops_wire_gf256_nib_vec4_16", "ops_wire_gf256_nib_vec4.comp", ["-DTILE_E=16"]),
    ("spv_ops_wire_gf256_nib_vec4_32", "ops_wire_gf256_nib_vec4.comp", ["-DTILE_E=32"]),
]


def compile_shader(src_path, flags, spv_path):
    if shutil.which("glslc"):
        cmd = [
            "glslc",
            "--target-env=vulkan1.2",
            "-fshader-stage=comp",
            "-O",
        ] + flags + [src_path, "-o", spv_path]
        subprocess.run(cmd, check=True)
    elif shutil.which("glslangValidator"):
        cmd = [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "comp",
        ] + flags + [src_path, "-o", spv_path]
        subprocess.run(cmd, check=True)
        if shutil.which("spirv-opt"):
            subprocess.run(["spirv-opt", "-O", spv_path, "-o", spv_path], check=True)
    else:
        raise RuntimeError("Neither glslc nor glslangValidator found in PATH")


def main():
    hdr = [
        "#ifndef VXFEC_VK_SPV_H",
        "#define VXFEC_VK_SPV_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
    ]
    src_out = [
        '#include "vk_spv.h"',
        "",
    ]
    for name, src, flags in SHADERS:
        src_path = os.path.join(HERE, src)
        with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tmp:
            spv_path = tmp.name
        try:
            compile_shader(src_path, flags, spv_path)
            with open(spv_path, "rb") as sf:
                data = sf.read()
            words = [
                int.from_bytes(data[i : i + 4], "little")
                for i in range(0, len(data), 4)
            ]
            hdr.append(f"extern const uint32_t {name}[];")
            hdr.append(f"extern const size_t {name}_bytes;")
            src_out.append(f"const uint32_t {name}[] = {{")
            for i in range(0, len(words), 6):
                chunk = ", ".join(f"0x{w:08x}u" for w in words[i : i + 6])
                src_out.append(f"    {chunk},")
            src_out.append("};")
            src_out.append(f"const size_t {name}_bytes = sizeof({name});")
            src_out.append("")
        finally:
            if os.path.exists(spv_path):
                os.unlink(spv_path)
    hdr.append("")
    hdr.append("#endif /* VXFEC_VK_SPV_H */")
    with open(OUT_HDR, "w") as f:
        f.write("\n".join(hdr) + "\n")
    with open(OUT_SRC, "w") as f:
        f.write("\n".join(src_out) + "\n")
    if shutil.which("clang-format"):
        subprocess.run(["clang-format", "-style=file", "-i", OUT_HDR, OUT_SRC], check=True)


if __name__ == "__main__":
    main()
