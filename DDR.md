# DDR

## DDR vs. SSD
- DDR的读写速度约为17ns，而SSD约为50us，速度快近3000x倍
- DDR为二维存储结构，SSD为三维
- DDR中单个memory cell只存储1bit信息，SSD可以存储3-4bit？
- 为什么DDR电荷泄露远快于SSD?
DDR的存储原件为电容，而SSD为浮栅晶体管。SSD电荷泄露慢的根本原因在于浮栅结构和隧穿效应，电荷被锁在一个高质量绝缘体构成的监狱中，逃逸难度高。通常DDR电荷在若干ms中泄露，而SSD可高达十年。
- 为什么DDR读写速度高于SSD？
    - 工作原理：电子开关 vs. 浮栅隧穿
        - DRAM：操作的是电容的充放电和MOSFET的开关，纯粹是电子移动，速度为纳秒级。
        - SSD：量子隧穿相对缓慢。同时由于一个单元需要存储约4位数据，需要控制16种电荷量，多次比较验证费时。
- DIMM: 8 DRAM Chips（a stick）
![alt text](image-1.png)
- DRAM 
![alt text](image.png)  
    - 数据线：32*2
    - 地址线：21*2
    - 控制线：7*2
    - 单个DRAM Chips单次输出8bit，1.7billion个memory cell
    - 单个DRAM由8*4个bank构成 
    ![alt text](image-2.png)
    65536根字线，8192根位线。 
    ![alt text](image-3.png)
    - 单次读写从8192列中一次性读写8个cell
    - 300mm的晶圆一次性生产2500个die
    ![alt text](image-4.png)
    - 1T1C DRAM Memory Cell
    ![alt text](image-6.png)
    - 三个操作
        - 读
        - 写
        - 刷新:每64ms刷新一次
    - 构成
        - 1T1C的2维阵列（包含字线和位线）
        - sense amplifier
        - column multiplexer
        - write driver
        - read driver
        - DRAM control
    - row hit & row miss
    尽可能提高row hit的比例有助于加速内存读写。将一整块DRAM分成32个subbank，每个subbank都有独立的字线，可以提高row hit的概率？ 
    ![alt text](image-7.png)
- DRAM vs. GRAM
![alt text](image-8.png)