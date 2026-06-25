---
sidebar_position: 2
title: Install cuDNN and TensorRT
description: Install NVIDIA cuDNN and TensorRT on Ubuntu 24.04
---

# Install cuDNN and TensorRT

```bash
sudo apt install tensorrt tensorrt-dev
```

Set the TensorRT environment variables system-wide via `/etc/profile.d/tensorrt.sh`:

```bash
sudo tee /etc/profile.d/tensorrt.sh << 'EOF'
# Runtime path — required to run TensorRT command-line tools
export PATH=/usr/src/tensorrt/bin${PATH:+:$PATH}
EOF
```

Re-login or run `source /etc/profile.d/tensorrt.sh` to apply immediately.
