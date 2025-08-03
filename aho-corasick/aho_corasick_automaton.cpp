#include "aho_corasick_automaton.h"

#include <algorithm>  // For std::max, std::sort, std::unique
#include <fstream>    // For std::ifstream
#include <iostream>   // For std::cerr, std::cout

// Constructor implementation
AhoCorasickAutomaton::AhoCorasickAutomaton() {
    // Initial allocation of space
    ac_base_.resize(1000, 0);
    ac_check_.resize(1000, NO_NODE);
    fail_link_.resize(1000, 0);
    output_patterns_.resize(1000);
    dictionary_match_link_.resize(1000, 0);
    ac_node_children_map_.resize(1000);

    // Initialize free list for the internal Trie
    for (uint32_t i = 0; i < ac_base_.size(); ++i) {
        ac_free_list_.insert(i);
    }

    // Initialize the root node (index 0) of the internal Trie
    ac_base_[0] = 1;         // Starting offset for the root node
    ac_check_[0] = 0;        // Check value for the root node
    ac_free_list_.erase(0);  // Root node is now occupied
}

// Helper function for internal Trie: checks if a position is used
bool AhoCorasickAutomaton::IsAcUsed(uint32_t pos) const {
    return ac_free_list_.find(pos) == ac_free_list_.end();
}

// Helper function for internal Trie: expands arrays
void AhoCorasickAutomaton::AcExpand(uint32_t expect_size) {
    if (ac_base_.size() >= expect_size) { return; }
    uint32_t old_size = ac_base_.size();
    ac_base_.resize(expect_size);
    ac_check_.resize(expect_size);
    fail_link_.resize(expect_size, 0);
    output_patterns_.resize(expect_size);
    dictionary_match_link_.resize(expect_size, 0);
    ac_node_children_map_.resize(expect_size);

    for (uint32_t i = old_size; i < ac_base_.size(); ++i) {
        ac_base_[i] = 0;
        ac_check_[i] = NO_NODE;
        ac_free_list_.insert(i);
        fail_link_[i] = 0;
        output_patterns_[i].clear();
        dictionary_match_link_[i] = 0;
        ac_node_children_map_[i].clear();
    }
    std::cerr << "DEBUG: AcExpand from " << old_size << " to " << expect_size << "\n";  // Debug output
}

// Helper function for internal Trie: finds a suitable base value and expands if
// necessary
int AhoCorasickAutomaton::AcCheckAndExpand(const std::vector<char32_t>& child_nodes_chars) {
    if (ac_free_list_.empty()) { AcExpand(ac_base_.size() * 2); }

    auto iter = ac_free_list_.begin();
    int add;
    while (true) {
        add = *iter;

        // Ensure 'add' itself is free
        if (IsAcUsed(add)) {
            ++iter;
            if (iter == ac_free_list_.end()) {
                AcExpand(ac_base_.size() * 2);
                iter = ac_free_list_.begin();
            }
            continue;
        }

        if (!child_nodes_chars.empty()) {
            uint32_t max_char_val = 0;
            for (char32_t c_val : child_nodes_chars) {
                if (static_cast<uint32_t>(c_val) > max_char_val) max_char_val = c_val;
            }
            uint32_t need_size = add + max_char_val + 1;
            if (need_size > ac_base_.size()) {
                AcExpand(need_size * 2);
                iter = ac_free_list_.find(add);
                if (iter == ac_free_list_.end()) {
                    iter = ac_free_list_.begin();
                    add = *iter;
                }
            }
        }

        bool success = true;
        for (char32_t child_char_val : child_nodes_chars) {
            if (IsAcUsed(add + static_cast<int>(child_char_val))) {
                success = false;
                break;
            }
        }

        if (success) {
            std::cerr << "DEBUG: AcCheckAndExpand found add = " << add << " for " << child_nodes_chars.size()
                      << " children.\n";  // Debug output
            return add;
        }

        ++iter;
        if (iter == ac_free_list_.end()) {
            AcExpand(ac_base_.size() * 2);
            iter = ac_free_list_.begin();
        }
    }
}

// Internal Trie insertion function, used to build the AC Trie structure
void AhoCorasickAutomaton::AcInsert(uint32_t father_pos, const std::vector<char32_t>& children_chars_to_insert) {
    // 1. Collect existing children and new children to insert
    std::map<char32_t, int> old_children_map_for_father;
    for (auto const& [char_val, child_idx] : ac_node_children_map_[father_pos]) {
        old_children_map_for_father[char_val] = child_idx;
    }

    std::vector<char32_t> all_children_chars = children_chars_to_insert;
    for (auto const& [char_val, child_idx] : old_children_map_for_father) {
        bool found = false;
        for (char32_t c : children_chars_to_insert) {
            if (c == char_val) {
                found = true;
                break;
            }
        }
        if (!found) { all_children_chars.push_back(char_val); }
    }
    std::sort(all_children_chars.begin(), all_children_chars.end());
    all_children_chars.erase(std::unique(all_children_chars.begin(), all_children_chars.end()),
                             all_children_chars.end());

    // 2. Find new 'add' value for father_pos
    int new_add = this->AcCheckAndExpand(all_children_chars);
    int old_add = ac_base_[father_pos];

    // 3. Update father_pos's base_ value
    ac_base_[father_pos] = new_add;
    std::cerr << "DEBUG: AcInsert father_pos = " << father_pos << ", old base = " << old_add
              << ", new base = " << new_add << "\n";  // Debug output

    // 4. Relocate existing children and insert new children
    ac_node_children_map_[father_pos].clear();

    for (char32_t child_char_val : all_children_chars) {
        int old_child_pos = NO_NODE;
        if (old_children_map_for_father.count(child_char_val)) {
            old_child_pos = old_children_map_for_father[child_char_val];
        }

        int new_child_pos = new_add + static_cast<int>(child_char_val);

        if (new_child_pos >= static_cast<int>(ac_base_.size())) { AcExpand(new_child_pos + 100); }

        if (old_child_pos != NO_NODE) {
            if (old_child_pos != new_child_pos) {
                ac_base_[new_child_pos] = ac_base_[old_child_pos];
                fail_link_[new_child_pos] = fail_link_[old_child_pos];
                output_patterns_[new_child_pos] = output_patterns_[old_child_pos];
                dictionary_match_link_[new_child_pos] = dictionary_match_link_[old_child_pos];

                ac_node_children_map_[new_child_pos] = ac_node_children_map_[old_child_pos];

                for (auto const& [grandchild_char_val, grandchild_old_pos] : ac_node_children_map_[old_child_pos]) {
                    if (grandchild_old_pos < static_cast<int>(ac_check_.size()) &&
                        ac_check_[grandchild_old_pos] == old_child_pos) {
                        ac_check_[grandchild_old_pos] = new_child_pos;
                        std::cerr << "DEBUG:     Updated grandchild " << grandchild_old_pos << " check from "
                                  << old_child_pos << " to " << new_child_pos << "\n";  // Debug output
                    }
                }

                ac_base_[old_child_pos] = 0;
                ac_check_[old_child_pos] = NO_NODE;
                fail_link_[old_child_pos] = 0;
                output_patterns_[old_child_pos].clear();
                dictionary_match_link_[old_child_pos] = 0;
                ac_node_children_map_[old_child_pos].clear();
                ac_free_list_.insert(old_child_pos);
                std::cerr << "DEBUG:     Relocated node " << old_child_pos << " to " << new_child_pos
                          << "\n";  // Debug output
            } else {
                std::cerr << "DEBUG:     Node " << old_child_pos << " stayed at same position " << new_child_pos
                          << "\n";  // Debug output
            }
        } else {
            ac_base_[new_child_pos] = 0;
            std::cerr << "DEBUG:     Created new child at " << new_child_pos << "\n";  // Debug output
        }

        ac_check_[new_child_pos] = father_pos;
        ac_free_list_.erase(new_child_pos);
        ac_node_children_map_[father_pos][child_char_val] = new_child_pos;
        std::cerr << "DEBUG:   Child char = " << static_cast<int>(child_char_val) << ", child_pos = " << new_child_pos
                  << ", check = " << ac_check_[new_child_pos] << "\n";  // Debug output
    }
    ac_free_list_.erase(new_add);
}

// Internal Trie helper function: gets the next state given current state and
// character Follows failure links until a valid transition is found or root is
// reached
int AhoCorasickAutomaton::AcGetNextState(int current_state, char32_t char_val) {
    // int original_state = current_state; // Kept for potential debug output
    while (true) {
        auto it = ac_node_children_map_[current_state].find(char_val);
        if (it != ac_node_children_map_[current_state].end()) {
            // std::cerr << "DEBUG: AcGetNextState from " << original_state << " with
            // char " << static_cast<int>(char_val) << " -> direct to " << it->second
            // << "\n"; // Debug output
            return it->second;
        }

        if (current_state == 0) {
            // std::cerr << "DEBUG: AcGetNextState from " << original_state << " with
            // char " << static_cast<int>(char_val) << " -> stay at root (0)\n"; //
            // Debug output
            return 0;
        }
        // std::cerr << "DEBUG: AcGetNextState from " << current_state << " with
        // char " << static_cast<int>(char_val) << " -> follow fail_link to " <<
        // fail_link_[current_state] << "\n"; // Debug output
        current_state = fail_link_[current_state];
    }
}

// Add a pattern to the Aho-Corasick automaton
void AhoCorasickAutomaton::add_pattern(const std::string& pattern_str, int pattern_id) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    Sentence pattern_sent = converter.from_bytes(pattern_str);

    std::cerr << "DEBUG: Adding pattern '" << pattern_str << "' (ID: " << pattern_id << ")\n";  // Debug output

    int current_node_idx = 0;

    for (size_t i = 0; i < pattern_sent.size(); ++i) {
        char32_t char_val = pattern_sent[i];
        int target_idx_candidate = ac_base_[current_node_idx] + static_cast<int>(char_val);

        std::cerr << "DEBUG:   Current node: " << current_node_idx << ", char_val: " << static_cast<int>(char_val)
                  << ", target_idx_candidate: " << target_idx_candidate << "\n";  // Debug output

        if (target_idx_candidate >= static_cast<int>(ac_check_.size()) ||
            ac_check_[target_idx_candidate] != current_node_idx) {
            std::cerr << "DEBUG:     Path not found, inserting new node.\n";  // Debug output
            std::vector<char32_t> children_for_insert;
            for (auto const& [c_val, child_idx] : ac_node_children_map_[current_node_idx]) {
                children_for_insert.push_back(c_val);
            }
            children_for_insert.push_back(char_val);
            std::sort(children_for_insert.begin(), children_for_insert.end());
            children_for_insert.erase(std::unique(children_for_insert.begin(), children_for_insert.end()),
                                      children_for_insert.end());

            AcInsert(current_node_idx, children_for_insert);
        }
        current_node_idx = ac_base_[current_node_idx] + static_cast<int>(char_val);
        std::cerr << "DEBUG:   Moved to node: " << current_node_idx << "\n";  // Debug output
    }

    output_patterns_[current_node_idx].push_back(pattern_id);
    pattern_id_to_length_[pattern_id] = static_cast<int>(pattern_sent.length());
    pattern_id_to_string_[pattern_id] = pattern_sent;
    std::cerr << "DEBUG: Pattern ID " << pattern_id << " marked at node " << current_node_idx << "\n";  // Debug output
}

// Build failure links for the Aho-Corasick automaton
void AhoCorasickAutomaton::build_failure_links() {
    std::queue<int> q;

    std::cerr << "DEBUG: Building failure links...\n";  // Debug output

    for (auto const& [char_val, child_idx] : ac_node_children_map_[0]) {
        q.push(child_idx);
        fail_link_[child_idx] = 0;
        std::cerr << "DEBUG:   Root child " << child_idx << " (char " << static_cast<int>(char_val)
                  << ") fail_link = 0\n";  // Debug output
    }

    while (!q.empty()) {
        int current_node_idx = q.front();
        q.pop();

        std::cerr << "DEBUG: Processing node " << current_node_idx << "\n";  // Debug output

        int fail_node_idx = fail_link_[current_node_idx];
        if (!output_patterns_[fail_node_idx].empty()) {
            dictionary_match_link_[current_node_idx] = fail_node_idx;
            std::cerr << "DEBUG:   Node " << current_node_idx << " dictionary_match_link = " << fail_node_idx
                      << " (from non-empty output)\n";  // Debug output
        } else {
            dictionary_match_link_[current_node_idx] = dictionary_match_link_[fail_node_idx];
            std::cerr << "DEBUG:   Node " << current_node_idx
                      << " dictionary_match_link = " << dictionary_match_link_[fail_node_idx]
                      << " (from fail_node's dictionary_match_link)\n";  // Debug output
        }

        for (auto const& [child_char_val, child_idx] : ac_node_children_map_[current_node_idx]) {
            fail_link_[child_idx] = AcGetNextState(fail_link_[current_node_idx], child_char_val);
            q.push(child_idx);
            std::cerr << "DEBUG:     Child " << child_idx << " (char " << static_cast<int>(child_char_val)
                      << ") fail_link = " << fail_link_[child_idx] << "\n";  // Debug output
        }
    }
    std::cerr << "DEBUG: Failure links built.\n";  // Debug output
}

// Search for all patterns in the given text
void AhoCorasickAutomaton::Search(const std::string& text_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    Sentence text_sent = converter.from_bytes(text_str);

    int current_state = 0;
    std::cout << "搜索文本: '" << text_str << "'\n";
    std::cerr << "DEBUG: Searching text '" << text_str << "' (u32string length: " << text_sent.length()
              << ")\n";  // Debug output

    for (int i = 0; i < static_cast<int>(text_sent.size()); ++i) {
        char32_t char_val = text_sent[i];

        std::cerr << "DEBUG:   Processing char at index " << i << ": '" << converter.to_bytes(Sentence(1, char_val))
                  << "' (U+" << std::hex << static_cast<uint32_t>(char_val) << std::dec
                  << "), current_state = " << current_state << "\n";  // Debug output
        current_state = AcGetNextState(current_state, char_val);
        std::cerr << "DEBUG:     New current_state = " << current_state << "\n";  // Debug output

        int temp_state = current_state;
        while (temp_state != 0) {
            std::cerr << "DEBUG:     Checking temp_state = " << temp_state << "\n";  // Debug output
            if (!output_patterns_[temp_state].empty()) {
                for (int pattern_id : output_patterns_[temp_state]) {
                    int pattern_length = pattern_id_to_length_[pattern_id];
                    int start_index = i - pattern_length + 1;
                    std::cout << "  模式 '" << converter.to_bytes(pattern_id_to_string_[pattern_id])
                              << "' (ID: " << pattern_id << ") 在索引 [" << start_index << ", " << i << "] 处找到。\n";
                    std::cerr << "DEBUG:       FOUND pattern ID " << pattern_id << " at [" << start_index << ", " << i
                              << "]\n";  // Debug output
                }
            }
            temp_state = dictionary_match_link_[temp_state];
        }
    }
}

// Load patterns from a dictionary file (one word per line)
bool AhoCorasickAutomaton::Load(const std::string& file_path) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        std::cerr << "错误: 无法打开文件 " << file_path << "\n";
        return false;
    }

    // Clear existing patterns and reset the automaton to a clean state
    // Resetting to initial size and values
    ac_base_.assign(1000, 0);
    ac_check_.assign(1000, NO_NODE);
    ac_free_list_.clear();
    for (uint32_t i = 0; i < ac_base_.size(); ++i) {
        ac_free_list_.insert(i);
    }
    ac_base_[0] = 1;
    ac_check_[0] = 0;
    ac_free_list_.erase(0);

    fail_link_.assign(1000, 0);
    output_patterns_.assign(1000, std::vector<int>());
    dictionary_match_link_.assign(1000, 0);
    ac_node_children_map_.assign(1000, std::map<char32_t, int>());

    pattern_id_to_length_.clear();
    pattern_id_to_string_.clear();

    std::string line;
    int pattern_id_counter = 1;  // Start pattern IDs from 1

    // std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter; //
    // Not needed here, add_pattern handles conversion

    while (std::getline(ifs, line)) {
        if (line.empty()) { continue; }
        // Each line is a word/pattern
        add_pattern(line, pattern_id_counter++);
    }

    if (pattern_id_counter == 1) {  // Only root was processed, no actual patterns added
        std::cerr << "警告: 文件 " << file_path << " 中没有词语。\n";
        return true;  // No words, but loaded successfully (just empty)
    }

    // After adding all patterns, build the failure links
    build_failure_links();

    return true;
}

// (Optional) For debugging and visualizing Trie structure information
void AhoCorasickAutomaton::print_ac_trie_info() {
    std::cerr << "\nDEBUG: AhoCorasickAutomaton Trie Info:\n";                     // Debug output
    std::cerr << "索引\tBase\tCheck\tFail\t输出模式\t字典匹配链接\t子节点映射\n";  // Debug
                                                                                   // output
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;

    int max_node_idx = 0;
    for (uint32_t i = 0; i < ac_base_.size(); ++i) {
        if (IsAcUsed(i)) { max_node_idx = std::max(max_node_idx, static_cast<int>(i)); }
    }
    int print_limit = std::min(max_node_idx + 10, static_cast<int>(ac_base_.size()));

    for (int i = 0; i < print_limit; ++i) {
        std::cerr << i << "\t" << ac_base_[i] << "\t" << ac_check_[i] << "\t" << fail_link_[i] << "\t[";
        for (int pid : output_patterns_[i]) {
            std::cerr << pid << " ";
        }
        std::cerr << "]\t" << dictionary_match_link_[i] << "\t{";
        bool first_child = true;
        for (auto const& [char_val, child_idx] : ac_node_children_map_[i]) {
            if (!first_child) { std::cerr << ", "; }
            std::cerr << "'" << converter.to_bytes(Sentence(1, char_val)) << "':" << child_idx;
            first_child = false;
        }
        std::cerr << "}\n";
    }
    std::cerr << "DEBUG: End AhoCorasickAutomaton Trie Info.\n";  // Debug output
}
