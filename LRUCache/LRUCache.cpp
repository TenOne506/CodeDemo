
#include <iostream>
#include <unordered_map>
struct Node{
    int key;
    int value;

    Node* prev;
    Node* next;
    Node()=default;
    Node(int k,int v):key(k),value(v){}
};

class LRUCache{
private:
    int cap;
    std::unordered_map<int, Node*> cache;
    Node* dyhead;

    void remove(Node* node){
        node->prev->next=node->next;
        node->next->prev = node->prev;
    }

    void appendhead(Node* node){
        node->next=dyhead->next;
        node->prev=dyhead;
        node->next->prev=node;
        node->prev->next=node;
    }
public:
    LRUCache(int othercap):cap(othercap){
        dyhead=new Node();
        dyhead->prev=dyhead;
        dyhead->next=dyhead;
    }

    int get(int key){
        auto iter=cache.find(key);
        if(iter==cache.end()){
            std::cout<<"something wrong"<<std::endl;
            return -1;
        }

        auto* node = iter->second;
        remove(node);
        appendhead(node);
        return node->value;
    }

    void set(int key,int value){
        auto iter= cache.find(key);
        if(iter!=cache.end()){
            auto* node = iter->second;
            node->value=value;
            remove(node);
            appendhead(node);
        }else{
            auto* node =new Node(key,value);
            appendhead(node);
        }

        if(cache.size() > cap){
            auto* delnode = dyhead->prev;
            remove(delnode);
            cache.erase(delnode->key);
            delete  delnode;
        }
    }
};