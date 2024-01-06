# InternLM-Chat-7B

1. #### 创建开发机

   ![image-20240106164938353](README.assets/image-20240106164938353.png)

2. ### SSH连接

   在pycham中进行ssh连接

   先上传公钥到InterStudio

   ![image-20240106165501630](README.assets/image-20240106165501630.png)

   在Pycharm中选择密钥对连接即可

   ![image-20240106165304888](README.assets/image-20240106165304888.png)

3. ### 配置环境

   - 查看conda环境

     <img src="README.assets/image-20240106165831776.png" alt="image-20240106165831776" style="zoom:50%;" />

   - 利用conda来clone一个环境

     这里的share就是首页看到的云盘

     ![image-20240106181721403](README.assets/image-20240106181721403.png)

     再查看env

     <img src="README.assets/image-20240106181741648.png" alt="image-20240106181741648" style="zoom:50%;" />

     激活fiv环境安装所需的依赖

     <img src="README.assets/image-20240106181939760.png" alt="image-20240106181939760" style="zoom:67%;" />

     复制模型

     ![image-20240106182413625](README.assets/image-20240106182413625.png)

4. ### 代码部分

   clone代码

   ![image-20240106191311764](README.assets/image-20240106191311764.png)

   编辑代码

   ![image-20240106191842540](README.assets/image-20240106191842540.png)

   加载预处理模型，提供promt模板

   ![image-20240106191920259](README.assets/image-20240106191920259.png)

   问它问题
   
   ![image-20240106192249397](README.assets/image-20240106192249397.png)

