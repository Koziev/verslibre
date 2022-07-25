# -*- coding: utf-8 -*-
# Автор: Гусев Илья    https://github.com/IlyaGusev/rupo
# Описание: Модуль для описания разметки по ударениям и слогам.

import json
import os
import xml.etree.ElementTree as etree
from enum import Enum, unique
import re
from collections import OrderedDict
from typing import Set, List, Dict, Tuple
import jsonpickle
import logging

from poetry.phonetic import Accents, rhymed


CYRRILIC_LOWER_VOWELS = "аоэиуыеёюя"
CYRRILIC_LOWER_CONSONANTS = "йцкнгшщзхъфвпрлджчсмтьб"
VOWELS = "aeiouAEIOUаоэиуыеёюяАОЭИУЫЕЁЮЯ"
CLOSED_SYLLABLE_CHARS = "рлймнРЛЙМН"


#from rupo.settings import HYPHEN_TOKENS
#HYPHEN_TOKENS = resource_filename(__name__, "data/hyphen-tokens.txt")

HYPHEN_TOKENS = [
"-нибудь",
"-либо",
"-как",
"-ка",
"-нибуть",
"-где",
"-чего",
"-таки",
"-что",
"-какие",
"-куда",
"-го",
"-на",
"-под",
"во-",
"в-",
"по-",
"кое-",
"из-",
"вот-",
"ну-",
"на-",
"ни-",
"ей-",
"ой-",
"эх-",
]




class Token:
    @unique
    class TokenType(Enum):
        """
        Тип токена.
        """
        UNKNOWN = -1
        WORD = 0
        PUNCTUATION = 1
        SPACE = 2
        ENDLINE = 3
        NUMBER = 4

        def __str__(self):
            return str(self.name)

        def __repr__(self):
            return self.__str__()

    def __init__(self, text: str, token_type: TokenType, begin: int, end: int):
        """
        :param text: исходный текст.
        :param token_type: тип токена.
        :param begin: начало позиции токена в тексте.
        :param end: конец позиции токена в тексте.
        """
        self.token_type = token_type
        self.begin = begin
        self.end = end
        self.text = text

    def __str__(self):
        return "'" + self.text + "'" + "|" + str(self.token_type) + " (" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.text == other.text and self.token_type == other.token_type


class Tokenizer(object):
    """
    Класс токенизации.
    """
    @staticmethod
    def tokenize(text: str, remove_punct=False, remove_unknown=False, replace_numbers=False) -> List[Token]:
        """
        Токенизация текстов на русском языке с учётом знаков препинания и слов с дефисами.

        :param text: исходный текст.
        :return: список токенов.
        """
        tokens = []
        punctuation = ".,?:;!—"
        begin = -1
        for i, ch in enumerate(text):
            if ch.isalpha() or ch == "-":
                if begin == -1:
                    begin = i
            else:
                if begin != -1:
                    tokens.append(Tokenizer.__form_token(text, begin, i))
                    begin = -1
                token_type = Token.TokenType.UNKNOWN
                if ch in punctuation:
                    token_type = Token.TokenType.PUNCTUATION
                elif ch == "\n":
                    token_type = Token.TokenType.ENDLINE
                elif ch == " ":
                    token_type = Token.TokenType.SPACE
                elif ch.isdigit():
                    token_type = Token.TokenType.NUMBER
                if len(tokens) != 0 and tokens[-1].token_type == token_type:
                    tokens[-1].text += ch
                    tokens[-1].end += 1
                else:
                    tokens.append(Token(ch, token_type, i, i + 1))
        if begin != -1:
            tokens.append(Tokenizer.__form_token(text, begin, len(text)))
        tokens = Tokenizer.__hyphen_map(tokens)
        if remove_punct:
            tokens = [token for token in tokens if token.token_type != Token.TokenType.PUNCTUATION]
        if remove_unknown:
            tokens = [token for token in tokens if token.token_type != Token.TokenType.UNKNOWN]
        if replace_numbers:
            for token in tokens:
                if token.token_type != Token.TokenType.NUMBER:
                    continue
                token.text = "ЧИСЛО"
                token.token_type = Token.TokenType.WORD
        return tokens

    @staticmethod
    def __form_token(text, begin, end):
        word = text[begin:end]
        if word != "-":
            return Token(word, Token.TokenType.WORD, begin, end)
        else:
            return Token("-", Token.TokenType.PUNCTUATION, begin, begin + 1)

    @staticmethod
    def __hyphen_map(tokens: List[Token]) -> List[Token]:
        """
        Слова из словаря оставляем с дефисом, остальные разделяем.

        :param tokens: токены.
        :return: токены после обработки.
        """
        new_tokens = []
        hyphen_tokens = Tokenizer.__get_hyphen_tokens()
        for token in tokens:
            if token.token_type != Token.TokenType.WORD:
                new_tokens.append(token)
                continue
            is_one_word = True
            if "-" in token.text:
                is_one_word = False
                for hyphen_token in hyphen_tokens:
                    if hyphen_token in token.text or token.text in hyphen_token:
                        is_one_word = True
            if is_one_word:
                new_tokens.append(token)
            else:
                texts = token.text.split("-")
                pos = token.begin
                for text in texts:
                    new_tokens.append(Token(text, Token.TokenType.WORD, pos, pos+len(text)))
                    pos += len(text) + 1
        return new_tokens

    @staticmethod
    def __get_hyphen_tokens():
        """
        :return: содержание словаря, в котором прописаны слова с дефисом.
        """
        #with open(HYPHEN_TOKENS, "r", encoding="utf-8") as file:
        #    hyphen_tokens = [token.strip() for token in file.readlines()]
        #    return hyphen_tokens
        return HYPHEN_TOKENS


class SentenceTokenizer(object):
    @staticmethod
    def tokenize(text: str) -> List[str]:
        m = re.split(r'(?<=[^А-ЯЁ].[^А-ЯЁ][.?!;]) +(?=[А-ЯЁ])', text)
        return m


# =============================================================================

def count_vowels(string):
    num_vowels = 0
    for char in string:
        if char in VOWELS:
            num_vowels += 1
    return num_vowels


def get_first_vowel_position(string):
    for i, ch in enumerate(string):
        if ch in VOWELS:
            return i
    return -1





class Annotation:
    """
    Класс аннотации.
    Содержит начальную и конечную позицию в тексте, а также текст аннотации.
    """
    def __init__(self, begin: int, end: int, text: str) -> None:
        self.begin = begin
        self.end = end
        self.text = text


class Syllable(Annotation):
    """
    Разметка слога. Включает в себя аннотацию и номер слога, а также ударение.
    Если ударение падает не на этот слог, -1.
    """
    def __init__(self, begin: int, end: int, number: int, text: str, stress: int=-1) -> None:
        super(Syllable, self).__init__(begin, end, text)
        self.number = number
        self.stress = stress

    def vowel(self) -> int:
        """
        :return: позиция гласной буквы этого слога в слове (с 0).
        """
        return get_first_vowel_position(self.text) + self.begin

    def from_dict(self, d: dict) -> 'Syllable':
        self.__dict__.update(d)
        if "accent" in self.__dict__:
            self.stress = self.__dict__["accent"]
        return self


def get_syllables(word: str) -> List[Syllable]:
    """
    Разделение слова на слоги.
    :param word: слово для разбивки на слоги.
    :return syllables: массив слогов слова.
    """
    syllables = []
    begin = 0
    number = 0

    # В случае наличия дефиса разбиваем слова на подслова, находим слоги в них, объединяем.
    if "-" in word:
        if word == "-":
            return [Syllable(0, 1, 0, word)]

        word_parts = word.split("-")
        word_syllables = []
        last_part_end = 0
        for part in word_parts:
            part_syllables = get_syllables(part)
            if len(part_syllables) == 0:
                continue
            for i in range(len(part_syllables)):
                part_syllables[i].begin += last_part_end
                part_syllables[i].end += last_part_end
                part_syllables[i].number += len(word_syllables)
            word_syllables += part_syllables
            last_part_end = part_syllables[-1].end + 1
        return word_syllables

    # Для слов или подслов, в которых нет дефиса.
    for i, ch in enumerate(word):
        if ch not in VOWELS:
            continue
        if i + 1 < len(word) - 1 and word[i + 1] in CLOSED_SYLLABLE_CHARS:
            if i + 2 < len(word) - 1 and word[i + 2] in "ьЬ":
                # Если после сонорного согласного идёт мягкий знак, заканчиваем на нём. ("бань-ка")
                end = i + 3
            elif i + 2 < len(word) - 1 and word[i + 2] not in VOWELS and \
                    (word[i + 2] not in CLOSED_SYLLABLE_CHARS or word[i + 1] == "й"):
                # Если после сонорного согласного не идёт гласная или другой сонорный согласный,
                # слог закрывается на этом согласном. ("май-ка")
                end = i + 2
            else:
                # Несмотря на наличие закрывающего согласного, заканчиваем на гласной.
                # ("со-ло", "да-нный", "пол-ный")
                end = i + 1
        else:
            # Если после гласной идёт не закрывающая согласная, заканчиваем на гласной. ("ко-гда")
            end = i + 1
        syllables.append(Syllable(begin, end, number, word[begin:end]))
        number += 1
        begin = end
    if get_first_vowel_position(word) != -1:
        # Добиваем последний слог до конца слова.
        syllables[-1] = Syllable(syllables[-1].begin, len(word), syllables[-1].number,
                                 word[syllables[-1].begin:len(word)])

    # 05.04.2022
    # обработка случая однобуквенных слов (предлоги, частицы) из согласной, которые дают 0 слогов.
    if len(syllables) == 0:
        syllables.append(Syllable(begin=0, end=1, number=0, text=word))

    return syllables




class Word(Annotation):
    """
    Разметка слова. Включает в себя аннотацию слова и его слоги.
    """
    def __init__(self, begin: int, end: int, text: str, syllables: List[Syllable]) -> None:
        super(Word, self).__init__(begin, end, text)
        self.syllables = syllables

    def count_stresses(self) -> int:
        """
        :return: количество ударений в слове.
        """
        return sum(syllable.stress != -1 for syllable in self.syllables)

    def stress(self) -> int:
        """
        :return: последнее ударение в слове, если нет, то -1.
        """
        stress = -1
        for syllable in self.syllables:
            if syllable.stress != -1:
                stress = syllable.stress
        return stress

    def get_stressed_syllables_numbers(self) -> List[int]:
        """
        :return: номера слогов, на которые падают ударения.
        """
        return [syllable.number for syllable in self.syllables if syllable.stress != -1]

    def get_stresses(self) -> Set[int]:
        """
        :return: все ударения.
        """
        stresses = set()
        for syllable in self.syllables:
            if syllable.stress != -1:
                stresses.add(syllable.stress)
        return stresses

    def set_stresses(self, stresses: List[int]) -> None:
        """
        Задать ударения, все остальные убираются.

        :param stresses: позиции ударения в слове.
        """
        for syllable in self.syllables:
            if syllable.vowel() in stresses:
                syllable.stress = syllable.vowel()
            else:
                syllable.stress = -1

    def get_short(self) -> str:
        """
        :return: слово в форме "текст"+"последнее ударение".
        """
        return self.text.lower() + str(self.stress())

    def from_dict(self, d: dict) -> 'Word':
        self.__dict__.update(d)
        syllables = d["syllables"]  # type: List[dict]
        self.syllables = [Syllable(0, 0, 0, "").from_dict(syllable) for syllable in syllables]
        return self

    def to_stressed_word(self):
        from rupo.stress.word import StressedWord, Stress
        return StressedWord(self.text, set([Stress(pos, Stress.Type.PRIMARY) for pos in self.get_stresses()]))

    def __hash__(self) -> int:
        """
        :return: хеш разметки.
        """
        return hash(self.get_short())


class Line(Annotation):
    """
    Разметка строки. Включает в себя аннотацию строки и её слова.
    """
    def __init__(self, begin: int, end: int, text: str, words: List[Word]) -> None:
        super(Line, self).__init__(begin, end, text)
        self.words = words

    def from_dict(self, d) -> 'Line':
        self.__dict__.update(d)
        words = d["words"]  # type: List[dict]
        self.words = [Word(0, 0, "", []).from_dict(word) for word in words]
        return self

    def count_vowels(self):
        num_vowels = 0
        for word in self.words:
            for syllable in word.syllables:
                if get_first_vowel_position(syllable.text) != -1:
                    num_vowels += 1
        return num_vowels


class Markup: #(CommonMixin):
    """
    Класс данных для разметки в целом с экспортом/импортом в XML и JSON.
    """
    def __init__(self, text: str=None, lines: List[Line]=None) -> None:
        self.text = text
        self.lines = lines
        self.version = 2

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def from_json(self, st) -> 'Markup':
        d = json.loads(st)
        return self.from_dict(d)

    def from_dict(self, d) -> 'Markup':
        self.__dict__.update(d)
        lines = d["lines"]  # type: List[dict]
        self.lines = [Line(0, 0, "", []).from_dict(line) for line in lines]
        return self

    def to_xml(self) -> str:
        """
        Экспорт в XML.

        :return self: строка в формате XML
        """
        pass #return dicttoxml(self.to_dict(), custom_root='markup', attr_type=False).decode('utf-8').replace("\n", "\\n")

    def from_xml(self, xml: str) -> 'Markup':
        """
        Импорт из XML.

        :param xml: XML-разметка
        :return self: получившийся объект Markup
        """
        root = etree.fromstring(xml)
        if root.find("version") is None or int(root.find("version").text) != self.version:
            raise TypeError("Другая версия разметки")
        lines_node = root.find("lines")
        lines = []
        for line_node in lines_node.findall("item"):
            words_node = line_node.find("words")
            words = []
            for word_node in words_node.findall("item"):
                syllables_node = word_node.find("syllables")
                syllables = []
                for syllable_node in syllables_node.findall("item"):
                    stress_node = syllable_node.find("accent") \
                        if syllable_node.find("accent") is not None \
                        else syllable_node.find("stress")
                    stress = int(stress_node.text)
                    syllables.append(Syllable(int(syllable_node.find("begin").text),
                                              int(syllable_node.find("end").text),
                                              int(syllable_node.find("number").text),
                                              syllable_node.find("text").text,
                                              stress))
                words.append(Word(int(word_node.find("begin").text), int(word_node.find("end").text),
                                  word_node.find("text").text, syllables))
            lines.append(Line(int(line_node.find("begin").text), int(line_node.find("end").text),
                              line_node.find("text").text, words))
        self.text = root.find("text").text.replace("\\n", "\n")
        self.lines = lines
        return self

    def from_raw(self, text: str) -> 'Markup':
        """
        Импорт из сырого текста с ударениями в конце слов

        :param text: текст.
        :return: разметка.
        """

        pos = 0
        lines = []
        for line in text.split("\n"):
            if line == "":
                continue
            line_tokens = []
            for word in line.split(" "):
                i = -1
                ch = word[i]
                stress = ""
                while ch.isdigit() or ch == "-":
                    stress += ch
                    i -= 1
                    ch = word[i]
                line_tokens.append((word[:i+1], int(stress[::-1])))
            words = []
            line_begin = pos
            for pair in line_tokens:
                token = pair[0]
                stress = pair[1]
                syllables = get_syllables(token)
                for j in range(len(syllables)):
                    syllables[j].begin += pos
                    syllables[j].end += pos
                word = Word(pos, pos + len(token), token, syllables)
                word.set_stresses([stress])
                words.append(word)
                pos += len(token) + 1
            lines.append(Line(line_begin, pos, " ".join([pair[0] for pair in line_tokens]), words))
        self.text = "\n".join([line.text for line in lines])
        self.lines = lines
        return self

    @staticmethod
    def process_text(text: str, stress_predictor) -> 'Markup':
        """
        Получение начального варианта разметки по слогам и ударениям.

        :param text: текст для разметки
        :param stress_predictor: предсказатель ударений.
        :return markup: разметка по слогам и ударениям
        """
        begin_line = 0
        lines = []
        words = []
        text_lines = text.split("\n")
        for text_line in text_lines:
            tokens = [token for token in Tokenizer.tokenize(text_line) if token.token_type == Token.TokenType.WORD]
            for token in tokens:
                word = Word(begin_line + token.begin, begin_line + token.end, token.text, get_syllables(token.text))
                # Проставляем ударения.
                stresses = stress_predictor.predict(token.text.lower())
                # Сопоставляем ударения слогам.
                if len(word.syllables) > 1:
                    word.set_stresses(stresses)
                words.append(word)
            end_line = begin_line + len(text_line)
            lines.append(Line(begin_line, end_line, text_line, words))
            words = []
            begin_line = end_line + 1
        return Markup(text, lines)


# ==================================================================================


class TreeNode:
    """
    Нода дерева разбора шаблона.
    """
    leaf_chars = "usUS"
    non_leaf_chars = "*?w"

    def __init__(self, parent: 'TreeNode', children: List['TreeNode'], text: str, pattern_pos: int):
        """
        :param parent: родитель ноды.
        :param children: дети ноды.
        :param text: символ, соответствующий ноде.
        :param pattern_pos: позиция символа в шаблоне
        """
        self.parent = parent  # type: TreeNode
        self.children = children  # type: List[TreeNode]
        self.text = text  # type: str
        self.pattern_pos = pattern_pos  # type: int

    def get_level(self) -> int:
        """
        :return: высота ноды в дереве.
        """
        parent = self.parent
        level = 0
        while parent is not None:
            parent = parent.parent
            level += 1
        return level

    def get_next_sibling(self) -> 'TreeNode':
        """
        :return: соседняя нода справа.
        """
        siblings = self.parent.children
        index = siblings.index(self) + 1
        if index < len(siblings):
            return siblings[index]
        return None

    def get_last_child_leaf(self) -> 'TreeNode':
        """
        :return: последняя нода изе детей, которая является листом.
        """
        for child in reversed(self.children):
            if child.is_leaf():
                return child
        return None

    def is_first_leaf(self) -> bool:
        if not self.is_leaf():
            return False
        return [child for child in self.parent.children if child.is_leaf()][0] == self

    def is_last_leaf(self) -> bool:
        if not self.is_leaf():
            return False
        return [child for child in self.parent.children if child.is_leaf()][-1] == self

    def get_most_left_leaf(self) -> 'TreeNode':
        """
        :return: самый левый потомок.
        """
        node = self
        while len(node.children) != 0:
            node = node.children[0]
        assert node.is_leaf()
        return node

    def print_tree(self) -> None:
        """
        Вывод дерева с корнем в этой ноде.
        """
        stack = list()
        stack.append(self)
        while len(stack) != 0:
            current_node = stack.pop()
            print("\t" * current_node.get_level(), current_node)
            stack += current_node.children

    def is_leaf(self) -> bool:
        """
        :return: является ли нода листом дерева.
        """
        return self.text in TreeNode.leaf_chars

    def __str__(self) -> str:
        return self.text + " " + str(self.pattern_pos)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.pattern_pos)

    def __eq__(self, other):
        return self.pattern_pos == other.pattern_pos


class State:
    """
    Состояние разбора.
    """

    def __init__(self, node: TreeNode, string_pos: int, strong_errors: int, weak_errors: int, pattern: str):
        """
        :param node: нода дерева, соответствующая состоянию.
        :param string_pos: позиция в сопоставляемой строке.
        :param strong_errors: количество ошибок в U и S.
        :param weak_errors: количество ошибок в u и s.
        :param pattern: шаблон - путь, до этого состояния.
        """
        self.node = node  # type: TreeNode
        self.string_pos = string_pos  # type: int
        self.strong_errors = strong_errors  # type: int
        self.weak_errors = weak_errors  # type: int
        self.pattern = pattern  # type: str

    def __str__(self) -> str:
        return str(self.node) + " " + str(self.string_pos) + " " + str(self.strong_errors) + " " + str(self.weak_errors)

    def __repr__(self) -> str:
        return self.__str__()


class PatternAnalyzer:
    """
    Сопоставлятель шаблона и строки.
    """

    def __init__(self, pattern: str, error_border: int = 8):
        """
        :param error_border: граница по ошибкам.
        :param pattern: шаблон.
        """
        self.pattern = pattern  # type: str
        self.tree = self.__build_tree(pattern)  # type: TreeNode
        self.error_border = error_border

    @staticmethod
    def count_errors(pattern: str, string: str, error_border: int = 8) -> Tuple[str, int, int, bool]:
        """
        :param pattern: шаблон.
        :param string: строка.
        :param error_border: граница по ошибкам.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок.
        """
        analyzer = PatternAnalyzer(pattern, error_border)
        return analyzer.__accept(string)

    @staticmethod
    def __build_tree(pattern: str) -> TreeNode:
        """
        Построение дерева шаблона.

        :param pattern: шаблон.
        :return: корень дерева.
        """
        root_node = TreeNode(None, list(), "R", -1)
        current_node = root_node
        for i, ch in enumerate(pattern):
            if ch == "(":
                node = TreeNode(current_node, list(), "()", i)
                current_node.children.append(node)
                current_node = node
            if ch == ")":
                node = current_node
                current_node = current_node.parent
                # Убираем бессмысленные скобки.
                if i + 1 < len(pattern) and pattern[i + 1] not in "*?":
                    current_node.children = current_node.children[:-1] + node.children
                    for child in node.children:
                        child.parent = current_node
            if ch in TreeNode.leaf_chars:
                current_node.children.append(TreeNode(current_node, list(), ch, i))
            # Заменяем скобки на нетерминалы.
            if ch in TreeNode.non_leaf_chars:
                current_node.children[-1].text = ch
                current_node.children[-1].pattern_pos = i
        return root_node

    def __accept(self, string: str) -> Tuple[str, int, int, bool]:
        """
        :param string: строка.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок, были ли ошибки.
        """
        current_states = [State(None, -1, 0, 0, "")]
        current_node = self.tree.get_most_left_leaf()
        for i, ch in enumerate(string):
            new_states = []
            for state in current_states:
                if state.node is not None:
                    current_node = self.__get_next_leaf(state.node)
                variants = self.__get_variants(current_node)

                # Каждый вариант - новое состояние.
                for variant in variants:
                    assert variant.is_leaf()
                    strong_errors = state.strong_errors + int(variant.text.isupper() and variant.text != ch)
                    weak_errors = state.weak_errors + int(variant.text.islower() and variant.text != ch.lower())
                    new_state = State(variant, i, strong_errors, weak_errors, state.pattern + variant.text)
                    if new_state.strong_errors + new_state.weak_errors > self.error_border:
                        continue
                    new_states.append(new_state)

            if len(new_states) == 0:
                # Можем закончить раньше, если по ошибкам порезали ветки, либо если шаблон меньше строки.
                current_states = PatternAnalyzer.__filter_states(current_states, self.tree)
                pattern, strong_errors, weak_errors = self.__get_min_errors_from_states(current_states)
                diff = (len(string) - i)
                return pattern, strong_errors + diff, weak_errors + diff, True

            current_states = new_states
        current_states = PatternAnalyzer.__filter_states(current_states, self.tree)
        return self.__get_min_errors_from_states(current_states) + (False,)

    @staticmethod
    def __get_variants(current_node: TreeNode) -> Set[TreeNode]:
        """
        :param current_node: текущая нода.
        :return: варианты ноды на том же символе строки, возникают из-за * и ? в шаблоне.
        """
        variants = set()
        current_variant = current_node
        while current_variant is not None:
            if current_variant not in variants:
                variants.add(current_variant)
            else:
                current_variant = current_variant.parent
            current_variant = PatternAnalyzer.__get_next_variant(current_variant)
        return variants

    @staticmethod
    def __get_next_variant(node: TreeNode) -> TreeNode:
        """
        Получение следующего варианта из варинатов текущей ноды.

        :param node: текущий вариант.
        :return: следующий вариант.
        """
        assert node.is_leaf()
        while node.parent is not None:
            parent = node.parent
            grandfather = parent.parent
            uncle = parent.get_next_sibling() if grandfather is not None else None
            is_variable = node.is_first_leaf() or not node.is_leaf()
            if is_variable and uncle is not None:
                return uncle.get_most_left_leaf()
            elif grandfather is not None and grandfather.text == "*" and grandfather.children[-1] == parent:
                return grandfather.get_most_left_leaf()
            if is_variable:
                node = parent
            else:
                break
        return None

    @staticmethod
    def __get_next_leaf(node: TreeNode) -> TreeNode:
        """
        Получение следующей ноды.

        :param node: текущая нода.
        :return: следующая нода.
        """
        assert node.is_leaf()
        while node.parent is not None:
            sibling = node.get_next_sibling()
            if sibling is not None:
                return sibling.get_most_left_leaf()
            elif node.parent.text == "*":
                return node.parent.get_most_left_leaf()
            node = node.parent
        return None

    @staticmethod
    def __filter_states(states: List[State], root: TreeNode) -> List[State]:
        """
        Фильтрация по наличию обязательных терминалов.

        :param states: состояния.
        :param root: корень дерева.
        :return: отфильтрованные состояния.
        """
        return [state for state in states if root.get_last_child_leaf() is None or
                state.node.pattern_pos >= root.get_last_child_leaf().pattern_pos]

    @staticmethod
    def __get_min_errors_from_states(states: List[State]) -> Tuple[str, int, int]:
        """
        :param states: состояния.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок.
        """
        if len(states) == 0:
            return "", 0, 0
        return min([(state.pattern, state.strong_errors, state.weak_errors) for i, state in enumerate(states)],
                   key=lambda x: (x[1], x[2], x[0]))


# =====================================================================================


class StressCorrection: #(CommonMixin):
    """
    Исправление ударения.
    """
    def __init__(self, line_number: int, word_number: int, syllable_number: int,
                 word_text: str, stress: int) -> None:
        """
        :param line_number: номер строки.
        :param word_number: номер слова.
        :param syllable_number: номер слога.
        :param word_text: текст слова.
        :param stress: позиция ударения (с 0).
        """
        self.line_number = line_number
        self.word_number = word_number
        self.syllable_number = syllable_number
        self.word_text = word_text
        self.stress = stress


class ClassificationResult: #(CommonMixin):
    """
    Результат классификации стихотворения по метру.
    """
    def __init__(self, count_lines: int=0) -> None:
        """
        :param count_lines: количество строк.
        """
        self.metre = None
        self.count_lines = count_lines
        self.errors_count = {k: 0 for k in MetreClassifier.metres.keys()}  # type: Dict[str, int]
        self.corrections = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]
        self.resolutions = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]
        self.additions = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]

    def get_metre_errors_count(self):
        """
        :return: получить количество ошибок на заданном метре.
        """
        return self.errors_count[self.metre]

    def to_json(self):
        """
        :return: сериализация в json.
        """
        return jsonpickle.encode(self)

    @staticmethod
    def str_corrections(collection: List[StressCorrection]) -> str:
        """
        :param collection: список исправлений.
        :return: его строковое представление.
        """
        return"\n".join([str((item.word_text, item.syllable_number)) for item in collection])

    def __str__(self):
        st = "Метр: " + str(self.metre) + "\n"
        st += "Снятая омография: \n" + ClassificationResult.str_corrections(self.resolutions[self.metre]) + "\n"
        st += "Неправильные ударения: \n" + ClassificationResult.str_corrections(self.corrections[self.metre]) + "\n"
        st += "Новые ударения: \n" + ClassificationResult.str_corrections(self.additions[self.metre]) + "\n"
        return st


class ErrorsTableRecord:
    def __init__(self, strong_errors, weak_errors, pattern, failed=False):
        self.strong_errors = strong_errors
        self.weak_errors = weak_errors
        self.pattern = pattern
        self.failed = failed

    def __str__(self):
        return self.pattern + " " + str(self.strong_errors) + " " + str(self.weak_errors)

    def __repr__(self):
        return self.__str__()


class ErrorsTable:
    def __init__(self, num_lines):
        self.data = {}
        self.num_lines = num_lines
        self.coef = OrderedDict(
            [("iambos", 0.3),
             ("choreios", 0.3),
             ("daktylos", 0.4),
             ("amphibrachys", 0.4),
             ("anapaistos", 0.4),
             ("dolnik3", 0.5),
             ("dolnik2", 0.5),
             ("taktovik3", 6.0),
             ("taktovik2", 6.0)
             ])
        self.sum_coef = OrderedDict(
            [("iambos", 0.0),
             ("choreios", 0.0),
             ("daktylos", 0.0),
             ("amphibrachys", 0.0),
             ("anapaistos", 0.0),
             ("dolnik3", 0.035),
             ("dolnik2", 0.035),
             ("taktovik3", 0.10),
             ("taktovik2", 0.10)
             ])
        for metre_name in MetreClassifier.metres.keys():
            self.data[metre_name] = [ErrorsTableRecord(0, 0, "") for _ in range(num_lines)]

    def add_record(self, metre_name, line_num, strong_errors, weak_errors, pattern, failed=False):
        self.data[metre_name][line_num] = ErrorsTableRecord(strong_errors, weak_errors, pattern, failed)

    def get_best_metre(self):
        for l in range(self.num_lines):
            strong_sum = 0
            weak_sum = 0
            for metre_name in self.data.keys():
                strong_sum += self.data[metre_name][l].strong_errors
                weak_sum += self.data[metre_name][l].weak_errors
            for metre_name, column in self.data.items():
                if strong_sum != 0:
                    column[l].strong_errors = column[l].strong_errors / float(strong_sum)
                if weak_sum != 0:
                    column[l].weak_errors = column[l].weak_errors / float(weak_sum)
        sums = dict()
        for metre_name in self.data.keys():
            sums[metre_name] = (0, 0)
        for metre_name, column in self.data.items():
            strong_sum = 0
            weak_sum = 0
            for l in range(self.num_lines):
                strong_sum += column[l].strong_errors
                weak_sum += column[l].weak_errors
            sums[metre_name] = (strong_sum, weak_sum)
        for metre_name, pair in sums.items():
            sums[metre_name] = self.sum_coef[metre_name] + (pair[0] + pair[1] / 2.0) * self.coef[metre_name] / self.num_lines
        logging.debug(sums)
        return min(sums, key=sums.get)


class MetreClassifier(object):
    """
    Классификатор, считает отклонения от стандартных шаблонов ритма(метров).
    """
    metres = OrderedDict(
        [("iambos", '(us)*(uS)(U)?(U)?'),
         ("choreios", '(su)*(S)(U)?(U)?'),
         ("daktylos", '(suu)*(S)(U)?(U)?'),
         ("amphibrachys", '(usu)*(uS)(U)?(U)?'),
         ("anapaistos",  '(uus)*(uuS)(U)?(U)?'),
         ("dolnik3", '(u)?(u)?((su)(u)?)*(S)(U)?(U)?'),
         ("dolnik2", '(u)?(u)?((s)(u)?)*(S)(U)?(U)?'),
         ("taktovik3", '(u)?(u)?((su)(u)?(u)?)*(S)(U)?(U)?'),
         ("taktovik2", '(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?')
         ])

    border_syllables_count = 20

    @staticmethod
    def classify_metre(markup):
        """
        Классифицируем стихотворный метр.

        :param markup: разметка.
        :return: результат классификации.
        """
        result = ClassificationResult(len(markup.lines))
        num_lines = len(markup.lines)
        errors_table = ErrorsTable(num_lines)
        for l, line in enumerate(markup.lines):
            for metre_name, metre_pattern in MetreClassifier.metres.items():
                line_syllables_count = sum([len(word.syllables) for word in line.words])

                # Строчки длиной больше border_syllables_count слогов не обрабатываем.
                if line_syllables_count > MetreClassifier.border_syllables_count or line_syllables_count == 0:
                    continue
                error_border = 7
                if metre_name == "dolnik2" or metre_name == "dolnik3":
                    error_border = 3
                if metre_name == "taktovik2" or metre_name == "taktovik3":
                    error_border = 2
                pattern, strong_errors, weak_errors, analysis_errored = \
                    PatternAnalyzer.count_errors(MetreClassifier.metres[metre_name],
                                                 MetreClassifier.__get_line_pattern(line),
                                                 error_border)
                if analysis_errored or len(pattern) == 0:
                    errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern, True)
                    continue
                corrections = MetreClassifier.__get_line_pattern_matching_corrections(line, l, pattern)[0]
                accentuation_errors = len(corrections)
                strong_errors += accentuation_errors
                errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
        result.metre = errors_table.get_best_metre()

        # Запомним все исправления.
        for l, line in enumerate(markup.lines):
            pattern = errors_table.data[result.metre][l].pattern
            failed = errors_table.data[result.metre][l].failed
            if failed or len(pattern) == 0:
                continue
            corrections, resolutions, additions =\
                MetreClassifier.__get_line_pattern_matching_corrections(line, l, pattern)
            result.corrections[result.metre] += corrections
            result.resolutions[result.metre] += resolutions
            result.additions[result.metre] += additions
            result.errors_count[result.metre] += len(corrections)
        return result

    @staticmethod
    def __get_line_pattern(line: Line) -> str:
        """
        Сопоставляем строку шаблону, считаем ошибки.

        :param line: строка.
        :return: количество ошибок
        """
        pattern = ""
        for w, word in enumerate(line.words):
            if len(word.syllables) == 0:
                pattern += "U"
            else:
                for syllable in word.syllables:
                    if syllable.stress != -1:
                        pattern += "S"
                    else:
                        pattern += "U"
        return pattern

    @staticmethod
    def __get_line_pattern_matching_corrections(line: Line, line_number: int, pattern: str) \
            -> Tuple[List[StressCorrection], List[StressCorrection], List[StressCorrection]]:
        """
        Ударения могут приходиться на слабое место,
        если безударный слог того же слова не попадает на икт. Иначе - ошибка.

        :param line: строка.
        :param line_number: номер строки.
        :param pattern: шаблон.
        :return: ошибки, дополнения и снятия
        """
        corrections = []
        resolutions = []
        additions = []
        number_in_pattern = 0
        for w, word in enumerate(line.words):
            # Игнорируем слова длиной меньше 2 слогов.
            if len(word.syllables) == 0:
                continue
            if len(word.syllables) == 1:
                if pattern[number_in_pattern].lower() == "s" and word.syllables[0].stress == -1:
                    additions.append(StressCorrection(line_number, w, 0, word.text, word.syllables[0].vowel()))
                number_in_pattern += len(word.syllables)
                continue
            stress_count = word.count_stresses()
            for syllable in word.syllables:
                if stress_count == 0 and pattern[number_in_pattern].lower() == "s":
                    # Ударений нет, ставим такое, какое подходит по метру. Возможно несколько.
                    additions.append(StressCorrection(line_number, w, syllable.number, word.text, syllable.vowel()))
                elif pattern[number_in_pattern].lower() == "u" and syllable.stress != -1:
                    # Ударение есть и оно падает на этот слог, при этом в шаблоне безударная позиция.
                    # Найдём такой слог, у которого в шаблоне ударная позиция. Это и есть наше исправление.
                    for other_syllable in word.syllables:
                        other_number_in_pattern = other_syllable.number - syllable.number + number_in_pattern
                        if syllable.number == other_syllable.number or pattern[other_number_in_pattern].lower() != "s":
                            continue
                        ac = StressCorrection(line_number, w, other_syllable.number, word.text, other_syllable.vowel())
                        if stress_count == 1 and other_syllable.stress == -1:
                            corrections.append(ac)
                        else:
                            resolutions.append(ac)
                number_in_pattern += 1
        return corrections, resolutions, additions

    @staticmethod
    def get_improved_markup(markup: Markup, result: ClassificationResult) -> Markup:
        """
        Улучшаем разметку после классификации метра.

        :param markup: начальная разметка.
        :param result: результат классификации.
        :return: улучшенная разметка.
        """
        for pos in result.corrections[result.metre] + result.resolutions[result.metre]:
            syllables = markup.lines[pos.line_number].words[pos.word_number].syllables
            for i, syllable in enumerate(syllables):
                syllable.stress = -1
                if syllable.number == pos.syllable_number:
                    syllable.stress = syllable.begin + get_first_vowel_position(syllable.text)
        for pos in result.additions[result.metre]:
            syllable = markup.lines[pos.line_number].words[pos.word_number].syllables[pos.syllable_number]
            syllable.stress = syllable.begin + get_first_vowel_position(syllable.text)

        return markup

    @staticmethod
    def improve_markup(markup: Markup) -> \
            Tuple[Markup, ClassificationResult]:
        """
        Улучшение разметки метрическим классификатором.

        :param markup: начальная разметка.
        """
        result = MetreClassifier.classify_metre(markup)
        improved_markup = MetreClassifier.get_improved_markup(markup, result)
        return improved_markup, result


class StressPredictorAdapter:
    def __init__(self, accentuator):
        self.accentuator = accentuator

    def predict(self, word):
        return [int(self.accentuator.predict_stressed_charpos(word.lower()))]


class MetreClassifierAdapter:
    def __init__(self, accentuator):
        self.stress_predictor = StressPredictorAdapter(accentuator)

    def predict(self, text):
        markup = Markup.process_text(text, self.stress_predictor)
        m = MetreClassifier.classify_metre(markup)
        return m


if __name__ == '__main__':
    tmp_dir = '../../tmp'
    accentuator = Accents()
    accentuator.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accentuator.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    mclassifier = MetreClassifierAdapter(accentuator)

    text = """Но есть только ты, полноводное чудо
Что не встречало меня на пути
И не дошло до меня, из ниоткуда
Туда, где твоё отражение найти"""

    m = mclassifier.predict(text)
    print(m)

