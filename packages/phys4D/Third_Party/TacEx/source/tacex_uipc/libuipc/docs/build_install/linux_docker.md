



Install Docker following the instructions.

[https:



Clone the repository with the following command:

```shell
git clone https:
```



```shell
cd libuipc
docker build -t pub/libuipc:dev -f artifacts/docker/dev.dockerfile .
```



```shell
docker run -it --rm --gpus all pub/libuipc:dev
```



You can run the `uipc_info.py` to check if the `Pyuipc` is installed correctly.

```shell
cd libuipc/python
python uipc_info.py
```

More samples are at [Pyuipc Samples](https: