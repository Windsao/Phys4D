

UPDATE 2024-12-14

- currently maintained in `xmake` branch
- build `xmake`
- contact @sailing-innocent for support



- fmt support: https:
- different heading style between cmake and xmake: https:





- xmake use a lot of process for parallel compilation in order to accelerate the compilation task. However, nvcc will consume a lot of memory, thus will eventually cause an OOM
- Solution:  set the multi-process manurally, e.g. `xmake -j8` to set 8 parallel compilation jobs
