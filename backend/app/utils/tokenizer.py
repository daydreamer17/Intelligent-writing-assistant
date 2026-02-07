"""
统一的文本分词工具

支持中文（jieba）和英文（简单分词）的混合文本处理
"""
from __future__ import annotations

import re
from typing import List, Set

# 尝试导入 jieba，如果没有安装则降级到简单分词
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


def _is_cjk(ch: str) -> bool:
    """检查字符是否为中日韩字符"""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or  # CJK Extension A
        0x20000 <= code <= 0x2A6DF or  # CJK Extension B
        0x2A700 <= code <= 0x2B73F or  # CJK Extension C
        0x2B740 <= code <= 0x2B81F or  # CJK Extension D
        0x2B820 <= code <= 0x2CEAF or  # CJK Extension E
        0xF900 <= code <= 0xFAFF or  # CJK Compatibility Ideographs
        0x2F800 <= code <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
    )


def _has_cjk(text: str) -> bool:
    """检查文本是否包含中日韩字符"""
    return any(_is_cjk(ch) for ch in text)


def tokenize(text: str, lowercase: bool = True) -> Set[str]:
    """
    文本分词（返回集合，用于关键字匹配）

    Args:
        text: 输入文本
        lowercase: 是否转换为小写

    Returns:
        分词结果集合
    """
    if not text:
        return set()

    # 如果包含中文且 jieba 可用，使用 jieba 分词
    if JIEBA_AVAILABLE and _has_cjk(text):
        tokens = jieba.cut(text)
        if lowercase:
            return {token.strip().lower() for token in tokens if token.strip()}
        else:
            return {token.strip() for token in tokens if token.strip()}

    # 否则使用简单的空格和标点分词
    if lowercase:
        text = text.lower()

    # 分离字母数字和其他字符
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            # 对于中文字符，单独作为一个 token
            if _is_cjk(ch):
                tokens.append(ch)

    if current:
        tokens.append("".join(current))

    return {token for token in tokens if token}


def tokenize_list(text: str, lowercase: bool = True) -> List[str]:
    """
    文本分词（返回列表，保留顺序和重复，用于向量化）

    Args:
        text: 输入文本
        lowercase: 是否转换为小写

    Returns:
        分词结果列表
    """
    if not text:
        return []

    # 如果包含中文且 jieba 可用，使用 jieba 分词
    if JIEBA_AVAILABLE and _has_cjk(text):
        tokens = jieba.cut(text)
        if lowercase:
            return [token.strip().lower() for token in tokens if token.strip()]
        else:
            return [token.strip() for token in tokens if token.strip()]

    # 否则使用简单的空格和标点分词
    if lowercase:
        text = text.lower()

    # 分离字母数字和其他字符
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            # 对于中文字符，单独作为一个 token
            if _is_cjk(ch):
                tokens.append(ch)

    if current:
        tokens.append("".join(current))

    return [token for token in tokens if token]


def tokenize_for_citation(text: str) -> List[str]:
    """
    用于引用匹配的分词（复用统一中英文分词逻辑）

    Args:
        text: 输入文本

    Returns:
        分词结果列表
    """
    return tokenize_list(text, lowercase=True)
