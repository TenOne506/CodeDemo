#include "aho_corasick_automaton.h"// Include the new Aho-Corasick header
#include <fstream>                 // For creating test dictionary file
#include <iostream>

// Note: The original DoubleArrayTrie class from the user's context is now fully
// defined within aho_corasick_automaton.cc. It's not explicitly needed here
// unless you want to create a separate instance of it.

// Main function to demonstrate Aho-Corasick functionality
int main() {
  // Create an AhoCorasickAutomaton instance
  AhoCorasickAutomaton ac_automaton;

  std::cout << "--- 测试手动添加模式 ---\n";
  std::cout << "添加模式中...\n";
  ac_automaton.add_pattern("he", 1);
  ac_automaton.add_pattern("she", 2);
  ac_automaton.add_pattern("his", 3);
  ac_automaton.add_pattern("hers", 4);
  ac_automaton.add_pattern("her", 5);
  ac_automaton.add_pattern("sh", 6);      // Add a short pattern
  ac_automaton.add_pattern("你好", 7);    // Add a Chinese pattern
  ac_automaton.add_pattern("世界", 8);    // Add another Chinese pattern
  ac_automaton.add_pattern("你好世界", 9);// Add a longer Chinese pattern

  std::cout << "\n构建失败链接中...\n";
  ac_automaton.build_failure_links();
  ac_automaton.print_ac_trie_info();// Print AC automaton structure info

  std::cout << "\n搜索文本: 'ahishers'\n";
  ac_automaton.Search("ahishers");

  std::cout << "\n搜索文本: 'ushers'\n";
  ac_automaton.Search("ushers");

  std::cout << "\n搜索文本: 'here'\n";
  ac_automaton.Search("here");

  std::cout << "\n搜索文本: 'mash'\n";
  ac_automaton.Search("mash");

  std::cout << "\n搜索文本: '你好世界你好'\n";
  ac_automaton.Search("你好世界你好");

  std::cout << "\n--- 测试从文件加载模式 ---\n";
  // Create a test dictionary file for Load function
  const std::string test_dict_file_name = "ac_patterns.txt";
  std::ofstream ofs(test_dict_file_name);
  if (ofs.is_open()) {
    ofs << "apple\n";
    ofs << "apply\n";
    ofs << "banana\n";
    ofs << "band\n";
    ofs << "你好\n";
    ofs << "世界\n";
    ofs << "你好世界\n";
    ofs.close();
    std::cout << "创建测试词典文件: " << test_dict_file_name << "\n";
  } else {
    std::cerr << "错误: 无法创建测试词典文件。\n";
    return 1;
  }

  // Create a new AhoCorasickAutomaton instance for file loading test
  AhoCorasickAutomaton ac_automaton_from_file;
  std::cout << "从文件 '" << test_dict_file_name << "' 加载模式...\n";
  if (ac_automaton_from_file.Load(test_dict_file_name)) {
    std::cout << "模式从文件加载成功，并构建了失败链接。\n";
    ac_automaton_from_file.print_ac_trie_info();// Print AC automaton structure
                                                // info after file load

    std::cout << "\n搜索文本 (文件加载后): 'i have an apple and a banana in "
                 "this world'\n";
    ac_automaton_from_file.Search("i have an apple and a banana in this world");

    std::cout << "\n搜索文本 (文件加载后): '你好世界你好，这是一个测试'\n";
    ac_automaton_from_file.Search("你好世界你好，这是一个测试");
  } else {
    std::cerr << "模式从文件加载失败。\n";
    return 1;
  }

  return 0;
}
