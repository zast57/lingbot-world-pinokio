module.exports = {
  run: [
    // Windows NVIDIA
    {
      when: "{{platform === 'win32' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
      }
    },
    // Windows no NVIDIA
    {
      when: "{{platform === 'win32' && gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio"
      }
    },
    // Linux NVIDIA: Detect CUDA architecture
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "fs.write",
      params: {
        path: "get_cuda_arch.py",
        json: false,
        text: "import ctypes, sys\ndef get_arch():\n    try:\n        if sys.platform == 'win32':\n            lib = ctypes.CDLL('nvcuda.dll')\n        else:\n            try:\n                lib = ctypes.CDLL('libcuda.so')\n            except:\n                lib = ctypes.CDLL('libcuda.so.1')\n        if lib.cuInit(0) != 0: return None\n        cnt = ctypes.c_int()\n        lib.cuDeviceGetCount(ctypes.byref(cnt))\n        if cnt.value == 0: return None\n        dev = ctypes.c_int()\n        lib.cuDeviceGet(ctypes.byref(dev), 0)\n        major = ctypes.c_int()\n        minor = ctypes.c_int()\n        lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)\n        lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)\n        print(f'CUDA_ARCH:{major.value}.{minor.value}')\n    except: pass\nget_arch()"
      }
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        message: "python get_cuda_arch.py",
        on: [{
          event: "/CUDA_ARCH:(\\d+\\.\\d+)/",
          kill: true
        }]
      }
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "local.set",
      params: {
        cuda_arch: "{{input.event ? input.event[1] : '8.9'}}"
      }
    },
    // Linux NVIDIA cu128 (Blackwell sm_120+)
    {
      "when": "{{gpu === 'nvidia' && platform === 'linux' && local.cuda_arch && Number(local.cuda_arch.split('.')[0]) >= 12}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 {{args && args.xformers ? 'xformers==0.0.30' : ''}} --index-url https://download.pytorch.org/whl/cu128 --force-reinstall",
          "{{args && args.triton ? 'uv pip install triton' : ''}}",
          "{{args && args.sageattention ? 'uv pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/sageattention-2.1.1%2Bcu128torch2.7.1-cp310-cp310-linux_x86_64.whl' : ''}}",
          "{{args && args.flashattention ? 'uv pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/flash_attn-2.8.3%2Bcu128torch2.7-cp310-cp310-linux_x86_64.whl' : ''}}"
        ]
      },
      "next": null
    },
    // Linux NVIDIA cu124 (older GPUs)
    {
      when: "{{platform === 'linux' && gpu === 'nvidia' && (!local.cuda_arch || Number(local.cuda_arch.split('.')[0]) < 12)}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
      }
    },
    // Linux AMD
    {
      when: "{{platform === 'linux' && gpu === 'amd'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
      }
    },
    // Linux CPU
    {
      when: "{{platform === 'linux' && !gpu}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
      }
    },
    // macOS
    {
      when: "{{platform === 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torchvision torchaudio"
      }
    }
  ]
}
