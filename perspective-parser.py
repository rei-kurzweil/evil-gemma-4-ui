class PerspectiveNode:
    def __init__(self, kind, marker=None):
        self.kind = kind
        self.marker = marker
        self.children = []
        self.text = ""
        self.closed = kind == "root"


class PerspectiveParser:
    def __init__(self):
        self.root = PerspectiveNode("root")
        self.stack = [self.root]

    def feed_sentence(self, sentence: str):
        cleaned = sentence.strip()
        if not cleaned:
            return

        if self._current_has_content():
            self._append_text(" ")

        self.feed_text(cleaned)

    def feed_text(self, text: str):
        index = 0
        while index < len(text):
            if text.startswith("**", index):
                self._toggle_emphasis("**")
                index += 2
                continue
            if text[index] == "*":
                self._toggle_emphasis("*")
                index += 1
                continue
            if text[index] == '"':
                self._toggle_quote()
                index += 1
                continue
            if text[index] == "(":
                self._open_context("parenthesis", "(")
                index += 1
                continue
            if text[index] == ")":
                if not self._close_context("parenthesis"):
                    self._append_text(")")
                index += 1
                continue

            self._append_text(text[index])
            index += 1

    def pretty_print(self):
        lines = []
        self._pretty_print_node(self.root, lines, 0)
        return "\n".join(lines)

    def _pretty_print_node(self, node, lines, depth):
        indent = "  " * depth
        if node.kind == "root":
            lines.append("root")
        elif node.kind == "text":
            lines.append(f"{indent}text: {node.text!r}")
        else:
            status = "" if node.closed else " [open]"
            marker = f" {node.marker}" if node.marker else ""
            lines.append(f"{indent}{node.kind}{marker}{status}")

        for child in node.children:
            self._pretty_print_node(child, lines, depth + 1)

    def _current_has_content(self):
        current = self.stack[-1]
        return bool(current.children or current.text)

    def _append_text(self, value: str):
        current = self.stack[-1]
        if current.children and current.children[-1].kind == "text":
            current.children[-1].text += value
            return

        node = PerspectiveNode("text")
        node.text = value
        current.children.append(node)

    def _toggle_emphasis(self, marker: str):
        kind = "bold" if marker == "**" else "italic"
        if self._top_matches(kind, marker):
            self._close_context(kind)
        else:
            self._open_context(kind, marker)

    def _toggle_quote(self):
        if self._top_matches("quote", '"'):
            self._close_context("quote")
        else:
            self._open_context("quote", '"')

    def _top_matches(self, kind: str, marker=None):
        if len(self.stack) <= 1:
            return False
        current = self.stack[-1]
        if current.kind != kind:
            return False
        if marker is not None and current.marker != marker:
            return False
        return True

    def _open_context(self, kind: str, marker=None):
        node = PerspectiveNode(kind, marker)
        node.closed = False
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def _close_context(self, kind: str):
        for index in range(len(self.stack) - 1, 0, -1):
            node = self.stack[index]
            if node.kind == kind:
                while len(self.stack) - 1 >= index:
                    closing = self.stack.pop()
                    closing.closed = closing.kind == kind
                return True
        return False
