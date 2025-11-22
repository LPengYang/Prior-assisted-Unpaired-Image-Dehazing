# Prior-assisted Unpaired Image Dehazing Framework for Enhanced Visibility in Real-world Hazy Scenarios

<hr />

> **Abstract:** *To facilitate a stable dehazing performance in real scenarios, this article proposes a novel prior-assisted unpaired image dehazing framework (PAUD), which obtains superior dehazing performance directly from real unpaired hazy/clear images. Specifically, a fast haze modulation (FHM) scheme is presented, which enables fast and flexible modulation in haze concentration for effortless production of diverse hazy samples, promoting the capability in dealing with complex scenarios. Moreover, an adaptive prior matching (APM) mechanism has been developed to alleviate the risk of misguidance caused by prior failure. This mechanism performs soft constraint with prior-based transmission by estimating a pixel-wise credibility map.
Extensive experiments demonstrate that the proposed method outperforms start-of-the-art methods in achieving enhanced visibility while requiring fewer parameters, providing effective and efficient visibility improvement under various hazy conditions.* 
<hr />


## 🎈 Quick Start on Pytorch 
```
python testing.py --model_load_path "your pretrained_model_path" --testing_image_dir "your testing_image_dir" --save_results_dir "your results save_dir"
```


## Framework & Architecture of Prior-assisted Unpaired Image Dehazing

![image](https://github.com/LPengYang/Prior-assisted-Unpaired-Image-Dehazing/blob/main/demonstration/Framework.png) 

![image](https://github.com/LPengYang/Prior-assisted-Unpaired-Image-Dehazing/blob/main/demonstration/Architecture.png) 



## Citation
If you use our work, please consider citing:

    @article{ling2025prior,
    title={Prior-assisted unpaired image dehazing framework for enhanced visibility in real-world hazy scenarios},
    author={Ling, Pengyang and Wang, Haoxuan and Chen, Huaian and Gu, Yuxuan and Jin, Yi and Zheng, Jinjin},
    journal={Expert Systems with Applications},
    volume={291},
    pages={128488},
    year={2025},
    publisher={Elsevier}}
