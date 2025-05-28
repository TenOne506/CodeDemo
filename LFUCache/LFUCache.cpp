#include <unordered_map>
class Node {
  public:
  int key;
  int value;
  int freq = 1;// 新书只读了一次
  Node *prev{};
  Node *next{};

  explicit Node(int otherk = 0, int otherv = 0) : key(otherk), value(otherv) {}
};

class LRUCache {
  private:
  int minfreq{};
  int cap;
  std::unordered_map<int, Node *> key2node;
  std::unordered_map<int, Node *> freq2head;
  explicit LRUCache(int rongliang) : cap(rongliang) {}
  void remove(Node *node) {
    node->next->prev = node->prev;
    node->prev->next = node->next;
  }
  Node *newList() {
    auto *dyhead = new Node();
    dyhead->next = dyhead;
    dyhead->prev = dyhead;
    return dyhead;
  }
  // 还是得考虑删除
  void push_front(int freq, Node *node) {
    // auto* dyhead = freq2head[freq];
    auto iter = freq2head.find(freq);
    if (iter == freq2head.end()) { iter = freq2head.emplace(freq, newList()).first; }
    auto *dyhead = iter->second;
    node->next = dyhead->next;
    node->prev = dyhead->prev;
    node->next->prev = node;
    node->prev->next = node;
  }

  Node *get_key(int key) {
    auto iter = key2node.find(key);
    if (iter == key2node.end()) { return nullptr; }

    auto *node = iter->second;
    remove(node);
    auto *dyhead = freq2head[node->freq];
    if (dyhead->prev == dyhead) {
      freq2head.erase(node->key);
      delete dyhead;
      if (minfreq == node->key) { minfreq++; }
    }

    push_front(++node->freq, node);
    return node;
  }

  int get(int key) {
    auto *node = get_key(key);
    return node != nullptr ? node->value : -1;
  }

  void put(int key, int value) {
    auto *node = get_key(key);
    if (node != nullptr) { node->value = value; }
    if (key2node.size() == this->cap) {
      auto *delhead = freq2head[minfreq];
      auto *delnode = delhead->prev;
      key2node.erase(delnode->key);
      remove(delnode);
      delete delnode;
      if (delhead->prev == delhead) {
        freq2head.erase(minfreq);
        delete delhead;
      }
    }

    key2node[key] = node = new Node(key, value);
    push_front(1, node);
    minfreq = 1;
  }
};
