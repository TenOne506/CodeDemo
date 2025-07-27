#include "aho_corasick_automaton.h"
#include <map>
#include <string>
#include <vector>
using Sentence = std::u32string;
class DoubleArrayTrie {
public:
  DoubleArrayTrie();
  ~DoubleArrayTrie();

private:
  void GenTrie(const std::vector<Sentence> &words);
  int CheckAndExpand(int next_pos);
  bool IsUsed(int cur_pos);

private:
  std::vector<int> base_;
  std::vector<int> check_;
  std::set<int> free_list_;
  std::vector<std::map<char32_t, int>> node_children_map_;
};
