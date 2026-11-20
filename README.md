e2 对应 Method 1：将明显无关的字段标记为 merge，其余字段标记为 keep。不使用 delete 操作。

e_clef_keep_or_delete 对应 Method 2：将明显无关的字段标记为 delete，其余字段标记为 keep。不使用 merge 操作。

e_jev 对应 Method 3：把 method1 中的分类器从Clef-flash换成JEV。

e_window 对应 Method 4：把 method1 中送入Clef-flash的上下文从完整交互历史换成窗口内的交互历史。
