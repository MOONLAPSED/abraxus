import inspect
import ast

def get_ast_breakdown(code):
    tree = ast.parse(code)
    visitor = ASTVisitor()
    visitor.visit(tree)
    return visitor.breakdown

class ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.breakdown = ""

    def visit_Assign(self, node):
        self.breakdown += f"Assignment: {node.targets[0].id} = {ast.dump(node.value)}\n"

    # Implement handlers for other AST node types as needed

# Example usage
code = "x = 1 + 2 * 3"
breakdown = get_ast_breakdown(code)
print(breakdown)


class DataClassAnalyzer:
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        output = f"{type(self.obj).__name__} (\n"
        for name, value in inspect.getmembers(self.obj):
            if not inspect.ismethod(value):  # Exclude methods
                output += f"  - {name}: {self._get_value_breakdown(value)}\n"
        output += ")"
        return output

    def _get_value_breakdown(self, value):
        value_type = type(value).__name__
        if isinstance(value, (list, tuple, set)):
            return f"[{' '.join([self._get_value_breakdown(item) for item in value])}]"
        elif isinstance(value, dict):
            return "{" + ", ".join(f"{k}: {self._get_value_breakdown(v)}" for k, v in value.items()) + "}"
        else:
            return value_type

# Example usage
class MyDataClass:
    def __init__(self, name, data):
        self.name = name
        self.data = data

data = MyDataClass("Example", [1, 2, {"key": "value"}])
analyzer = DataClassAnalyzer(data)
print(analyzer)