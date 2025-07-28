#ifndef AHO_CORASICK_AUTOMATON_H_
#define AHO_CORASICK_AUTOMATON_H_

#include <codecvt>  // For std::wstring_convert and std::codecvt_utf8
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

// Define Sentence type for Unicode characters
typedef std::u32string Sentence;

// Define a special value for empty or invalid nodes
const int NO_NODE = -1;

// AhoCorasickAutomaton class for implementing Aho-Corasick automaton
// functionality
class AhoCorasickAutomaton {
 public:
    // Internal Trie structure for the Aho-Corasick automaton, using arrays
    // similar to DoubleArrayTrie
    std::vector<int> ac_base_;
    std::vector<int> ac_check_;
    std::set<uint32_t> ac_free_list_;                            // Free list for the internal Trie
    std::vector<std::map<char32_t, int>> ac_node_children_map_;  // Child mapping for the internal Trie, key is
                                                                 // char32_t

    // Aho-Corasick specific members
    std::vector<int> fail_link_;
    std::vector<std::vector<int>> output_patterns_;  // Stores IDs of patterns ending at this node
    std::vector<int> dictionary_match_link_;         // Link for output propagation

    std::map<int, int> pattern_id_to_length_;             // Map from pattern ID to pattern length
    std::map<int, std::u32string> pattern_id_to_string_;  // Map from pattern ID to pattern string
                                                          // (u32string)

    // Constructor
    AhoCorasickAutomaton();

    // Add a pattern to the Aho-Corasick automaton
    void add_pattern(const std::string& pattern_str, int pattern_id);

    // Build failure links for the Aho-Corasick automaton
    void build_failure_links();

    // Search for all patterns in the given text
    void Search(const std::string& text_str);

    // Load patterns from a dictionary file (one word per line)
    bool Load(const std::string& file_path);

    // (Optional) For debugging and visualizing Trie structure information
    void print_ac_trie_info();

 private:
    // Helper function for internal Trie: checks if a position is used
    bool IsAcUsed(uint32_t pos) const;

    // Helper function for internal Trie: expands arrays
    void AcExpand(uint32_t expect_size);

    // Helper function for internal Trie: finds a suitable base value and expands
    // if necessary
    int AcCheckAndExpand(const std::vector<char32_t>& child_nodes_chars);

    // Internal Trie insertion function, used to build the AC Trie structure
    void AcInsert(uint32_t father_pos, const std::vector<char32_t>& children_chars_to_insert);

    // Helper function for internal Trie: gets the next state given current state
    // and character Follows failure links until a valid transition is found or
    // root is reached
    int AcGetNextState(int current_state, char32_t char_val);
};

#endif  // AHO_CORASICK_AUTOMATON_H_
