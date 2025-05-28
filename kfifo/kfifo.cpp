// example 6
#include <algorithm>
#include <cstring>
#include <iostream>
// 用数组实现的环型队列
class FIFO {
  static const unsigned int CAPACITY = 1024;// 容量：需要满足是2^N

  unsigned char buffer[CAPACITY];// 保存数据的缓冲区
  unsigned int in = 0;           // 写入位置
  unsigned int out = 0;          // 读取位置

  unsigned int free_space() const { return CAPACITY - in + out; }

  public:
  // 返回实际写入的数据长度（<= len），返回小于len时对应空闲空间不足
  unsigned int put(unsigned char *src, unsigned int len) {
    // 计算实际可写入数据长度（<=len）
    len = std::min(len, free_space());

    // 计算从in位置到buffer结尾有多少空闲空间
    unsigned int l = std::min(len, CAPACITY - (in & (CAPACITY - 1)));
    // 1. 把数据放入buffer的in开始的缓冲区，最多到buffer结尾
    memcpy(buffer + (in & (CAPACITY - 1)), src, l);
    // 2. 把数据放入buffer开头（如果上一步还没有放完），len -
    // l为0代表上一步完成数据写入
    memcpy(buffer, src + l, len - l);

    in += len;// 修改in位置，累加，到达uint32_max后溢出回绕
    return len;
  }

  // 返回实际读取的数据长度（<= len），返回小于len时对应buffer数据不够
  unsigned int get(unsigned char *dst, unsigned int len) {
    // 计算实际可读取的数据长度
    len = std::min(len, in - out);

    unsigned int l = std::min(len, CAPACITY - (out & (CAPACITY - 1)));
    // 1. 从out位置开始拷贝数据到dst，最多拷贝到buffer结尾
    memcpy(dst, buffer + (out & (CAPACITY - 1)), l);
    // 2. 从buffer开头继续拷贝数据（如果上一步还没拷贝完），len -
    // l为0代表上一步完成数据获取
    memcpy(dst + l, buffer, len - l);

    out += len;// 修改out，累加，到达uint32_max后溢出回绕
    return len;
  }
};