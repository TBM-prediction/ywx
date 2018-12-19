# File structure
The data is assumed to be placed as following in order for the notebooks to run.
```
.
├── tbmData
│   ├── CREC188
│   │   ├── device_data
│   │   ├── device_data_zip -> 设备数据
│   │   ├── geology_and_others -> 地质及其他信息(中铁装备)
│   │   ├── NDA -> 保密承诺书
│   │   ├── rules -> 竞赛说明及过程文件
│   │   ├── 保密承诺书
│   │   ├── 地质及其他信息(中铁装备)
│   │   ├── 竞赛说明及过程文件
│   │   └── 设备数据
│   └── tbmData.tar.xz
...
```

# Contribution note
Please ensure notebooks are properly striped before submitting a pull request. See [fastai docs](https://docs.fast.ai/dev/develop.html) 
for detailed instructions. (Hint: the tool used for striping notebooks is [nbstripout](https://github.com/kynan/nbstripout))

