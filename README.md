# HGCL_Rechorus
A HGCL model in Rechorus

我们在原有的Rechorus框架的基础上结合HGCL论文的代码实现了自己的HGCL类和HGCLReader类，使其能够在Rechorus框架上复现HGCL模型。

将HGCL.py放至Rechorus框架中的.\ReChorus-master\src\models\context目录下；
将HGCLReader.py放至Rechorus框架中的.\ReChorus-master\src\helpers目录下；
在.\ReChorus-master的终端下运行python main.py --model_name HGCL --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food即可。
