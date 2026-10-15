# Kaggle Docker Image Execution Without Docker Engine

## 1. Executive Summary

The active Kaggle notebook can execute Docker Hub images despite lacking a native Docker Engine. Two working routes were tested:

1. **Direct `udocker`** pulls and executes Docker images using a userspace runtime. It successfully ran `hello-world`, `alpine:3.22`, `python:3.13-alpine`, and `ubuntu:24.04`; a host-directory volume mount from `/kaggle/working` into the image was also verified.
2. **`udockerd`** exposes a Docker Engine–compatible HTTP API backed by `udocker`. The API was verified with image pulls, container create/start, log retrieval, and the Python Docker SDK. Basic execution worked for Alpine, Python Alpine, and Ubuntu images.

This is **not native Docker**. These approaches run image filesystems and processes through userspace techniques, not Docker's kernel-level isolation. Compatibility is incomplete and workload-dependent. Rootless Docker/Podman and other namespace-dependent runtimes are blocked by the current notebook sandbox.

## 2. Environment & Restrictions

Observed in the active remote notebook on 2026-10-11:

- OS: Ubuntu 24.04.5 LTS.
- Kernel: Linux 6.18.48+; architecture: x86_64.
- Python: 3.13.16.
- Host RAM reported by the notebook runner: approximately 31.3 GiB available at inspection time.
- `/kaggle/working`: approximately 20 GiB filesystem.
- The notebook process reports UID 0, but that does **not** mean it has all host/kernel privileges.
- No `docker`, `podman`, `containerd`, `nerdctl`, `runc`, `crun`, `apptainer`, or `singularity` executable was initially present.
- No Docker socket (`/var/run/docker.sock`), Podman socket, or `/dev/fuse` was found.
- cgroup v2 is mounted read-only.
- seccomp is enabled.
- `unshare -Ur id` fails with `Operation not permitted`.

### Consequence

Installing a CLI binary alone does not provide a native container engine. Rootless OCI runtimes need kernel features that this sandbox does not expose. Docker-in-Docker, native Podman, and typical rootless Apptainer/Singularity setups are therefore not practical with the observed restrictions.

## 3. Option A — Direct `udocker` (Recommended for Actual Workloads)

### Installation

In this notebook, the effective non-root account is `ubuntu`. The `udocker` CLI refuses to run as root, so install and run it as that account. The commands below reflect the tested paths:

```bash
runuser -u ubuntu -- env \
  HOME=/home/ubuntu \
  PYTHONUSERBASE=/home/ubuntu/.local \
  PIP_CONFIG_FILE=/dev/null \
  PIP_LOG=/home/ubuntu/pip.log \
  python3 -m pip install --user --disable-pip-version-check udocker

cd /home/ubuntu
runuser -u ubuntu -- env \
  HOME=/home/ubuntu \
  PYTHONUSERBASE=/home/ubuntu/.local \
  /home/ubuntu/.local/bin/udocker install
```

The `PYTHONUSERBASE` and `HOME` assignments matter in this environment; inherited pip settings initially tried to install under `/root/.local`. Also set the working directory to a path accessible to `ubuntu`: running from `/root` caused a `proot` chdir permission error.

### Verified image runs

```bash
cd /home/ubuntu
runuser -u ubuntu -- env HOME=/home/ubuntu PYTHONUSERBASE=/home/ubuntu/.local \
  /home/ubuntu/.local/bin/udocker run hello-world

runuser -u ubuntu -- env HOME=/home/ubuntu PYTHONUSERBASE=/home/ubuntu/.local \
  /home/ubuntu/.local/bin/udocker run alpine:3.22 \
  /bin/sh -c 'cat /etc/alpine-release; echo IMAGE_EXECUTION_OK'
```

Observed output included:

- `Hello from Docker!` from the official `hello-world` image.
- `3.22.6` and `IMAGE_EXECUTION_OK` from Alpine.

The following was also verified:

- `python:3.13-alpine` executed Python 3.13.16 and printed a success marker.
- `ubuntu:24.04` executed shell commands and reported Ubuntu 24.04.5 and glibc 2.39.

### Verified Kaggle working-directory mount

A host directory mount was successfully tested with:

```bash
cd /home/ubuntu
runuser -u ubuntu -- env HOME=/home/ubuntu PYTHONUSERBASE=/home/ubuntu/.local \
  /home/ubuntu/.local/bin/udocker run \
  --nobanner --rm \
  --volume=/kaggle/working:/workspace \
  alpine:3.22 \
  /bin/sh -c 'echo KAGGLE_WORKING_BIND_OK; ls /workspace | head'
```

The output included `KAGGLE_WORKING_BIND_OK` and existing notebook files, proving that `/kaggle/working` was visible inside the image at `/workspace`.

Direct `udocker` therefore provides the strongest verified route for workloads needing image execution plus access to Kaggle artifacts.

### Practical notes

- Use `--rm` for ephemeral containers when persistent container state is unnecessary.
- Use `--volume=/host/path:/container/path` for file sharing.
- Set `--workdir=...` if the application expects a specific working directory.
- Pulling/running a tag for the first time requires network access and downloads image layers.
- Dockerfile builds are not equivalent to running images. `udocker` is not a full build engine; build the image elsewhere and publish/pull it, or translate the build steps to a compatible userspace workflow.

## 4. Option B — `udockerd` Docker Engine–Compatible API

### What it provides

`udockerd` 0.3.5 was installed for the `ubuntu` account alongside `udocker` 1.3.17. It advertises Docker Engine API compatibility and runs via proot/fakechroot without root, namespaces, or cgroups.

The daemon was started as a background process on loopback:

```bash
cd /home/ubuntu
runuser -u ubuntu -- env HOME=/home/ubuntu PYTHONUSERBASE=/home/ubuntu/.local \
  /home/ubuntu/.local/bin/udockerd --host 127.0.0.1 --port 2375
```

In the active test session it is listening on `127.0.0.1:2375`. It is bound to loopback intentionally; do not expose an unauthenticated Docker-compatible API on a public interface.

### API checks performed

- `GET /_ping` returned HTTP 200 and `OK`.
- `GET /version` reported API version 1.41.
- `POST /images/create?fromImage=alpine&tag=3.22` downloaded the image.
- The image list reflected `alpine:3.22` and `hello-world:latest`.
- Container create and start endpoints returned HTTP 201 and 204 respectively.
- Container logs returned the tested marker `DOCKER_API_IMAGE_RUN_OK` and Alpine version `3.22.6`.
- The official Python Docker SDK connected and listed images/containers.
- The SDK successfully ran `python:3.13-alpine` and `ubuntu:24.04` with expected output.
- The standard `DOCKER_HOST` environment variable worked:

```bash
DOCKER_HOST=tcp://127.0.0.1:2375 python3 -c '
import docker
client = docker.from_env()
print("API version:", client.version()["ApiVersion"])
output = client.containers.run(
    "python:3.13-alpine",
    ["python", "-c", "print(\\"PYTHON_IMAGE_OK\\")"],
    remove=True,
)
print(output.decode())
'
```

This prints API version `1.41` and `PYTHON_IMAGE_OK` in the verified environment.

### Critical limitation — bind mounts through the API

The Python Docker SDK was used to request a bind mount of a host fixture file into an Alpine container. The container started, but the requested file did not appear at the destination. Thus **Docker-API volume/bind-mount behavior is not validated and failed in this test**.

Use direct `udocker --volume=...` instead when the workload must read or write notebook files. Do not assume that `udockerd` implements all Docker options just because basic API endpoints work.

## 5. Option C — `proot` With a Separate Root Filesystem

`proot` 5.1.0 was installed from Ubuntu packages and tested with a separate Alpine 3.24.2 minirootfs.

The minimal root filesystem was extracted under:

`/kaggle/working/proot-test/rootfs`

The command below worked:

```bash
proot -0 \
  -r /kaggle/working/proot-test/rootfs \
  -w / \
  /bin/sh -c 'cat /etc/alpine-release; id; /bin/busybox echo proot-container-ok'
```

It printed Alpine `3.24.2` and `proot-container-ok`.

This option demonstrates that userspace separation works, but starting from a rootfs manually is less convenient than pulling image layers through `udocker`. For Docker Hub images, prefer direct `udocker` unless a custom rootfs is specifically useful.

## 6. Options Tested or Ruled Out

| Option | Observed result | Conclusion |
|---|---|---|
| Docker CLI / Engine | No executable or daemon socket initially present | No native engine available |
| Podman | No executable or service socket initially present | Not available out of the box |
| containerd / nerdctl / runc / crun | No binaries found initially | Not available out of the box |
| Rootless namespaces | `unshare -Ur id` fails with `Operation not permitted` | Rootless OCI runtimes blocked by current sandbox |
| Apptainer / Singularity | Not installed; common unprivileged route depends on restricted kernel features | Not a promising route under observed restrictions |
| `/dev/fuse`-based solutions | `/dev/fuse` absent | FUSE-dependent option unavailable without additional device support |
| `udocker` | Direct Docker Hub image execution verified, including file mounts | Recommended image runner |
| `udockerd` | Docker API pull/create/start/log and Python SDK verified | Useful API shim, incomplete compatibility |
| `proot` + extracted rootfs | A separate Alpine rootfs executed successfully | Works for manual userspace environments |

The checks only describe the notebook sandbox at the time of inspection. Another Kaggle image, runtime configuration, or future policy change may expose different kernel capabilities.

## 7. Limitations and Expectations

Neither `udocker` nor `udockerd` is a drop-in replacement for Docker Engine in every workload.

Do not assume availability of:

- Kernel-level container security/isolation comparable to Docker.
- Independent network namespaces, full Docker networking, privileged containers, or cgroup-based resource limits.
- Full Docker Compose or BuildKit behavior.
- Exact semantics for all Docker Engine API endpoints, volume handling, or device passthrough.
- Compatibility with applications requiring systemd, kernel modules, mount privileges, nested runtimes, or namespace creation.

A successful image run means that the tested userspace process executed, not that the image is fully isolated from the host. Avoid running untrusted images: userspace emulation is not a security boundary.

GPU access, port publishing, shared memory, and device support should be tested for the specific target image rather than inferred from the CPU-only smoke tests documented here.

## 8. Recommendation

1. **Use direct `udocker` first** for pulling and running Docker Hub images, especially when the image needs access to `/kaggle/working`.
2. **Use `udockerd`** when existing code expects a Docker Engine API or Python Docker SDK, while keeping in mind the observed bind-mount limitation and broader feature gaps.
3. **Use native Docker on an external host or a suitable VM** when you need real kernel isolation, Dockerfile builds, Compose, or precise Docker networking/volume semantics.

The most important result is that Docker images can run on this notebook without native Docker Engine. The verified working path is `udocker`; `udockerd` also gives a useful but partial Docker-compatible API.
