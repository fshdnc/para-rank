### Getting the avg of three datapoints
```
In [10]: s="""0.0538^I0.0554^I0.0423^I0.0336^I0.5324
    ...: 0.0438^I0.0449^I0.0328^I0.0258^I0.5797
    ...: 0.0352^I0.0363^I0.0248^I0.0158^I0.5728
    ...: 0.0193^I0.0118^I0.0079^I0.0037^I0.6211
    ...: 0.0155^I0.0121^I0.0071^I0.0034^I0.6188
    ...: 0.0147^I0.0114^I0.0073^I0.0033^I0.6141
    ...: 0.0209^I0.0099^I0.0053^I0.0017^I0.6306
    ...: 0.0147^I0.0066^I0.0031^I0.0009^I0.6501
    ...: 0.021^I0.0098^I0.006^I0.0016^I0.637
    ...: 0.0176^I0.0098^I0.0045^I0.0013^I0.619
    ...: 0.0294^I0.0194^I0.0092^I0.0034^I0.5829
    ...: 0.0233^I0.0077^I0.0039^I0.0018^I0.6188""".split("\n")

In [11]: import numpy as np
    ...: for i in range(int(len(s)/3)):
    ...:     points =s[i*3:i*3+3]
    ...:     points = [p.split("\t") for p in points]
    ...:     avgs = []
    ...:     for j in range(len(points[0])):
    ...:         avgs.append(str(np.round(np.mean([float(p[j]) for p in points]),4)))
    ...:     print("\t".join(avgs))

```

2022.5.13 All files used for debugging deleted. The problem was that PyTorch module 1.3.1 should not have been used. Use 1.10