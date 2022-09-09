# import os
# import re
# from numbers import Number
#
#
# class Declarable:
#     """
#     Subclasses of this class can be specified by their declarations.
#     Each declaration uniquely corresponds to a filename.
#     Allowed class parameters are: primitives (bool, int, float, str, tuple, list, set, dict) or Declarable.
#     In order to pass a complex object to class constructor, it must extend Declarable or try to use
#     primitive types (e.g. class name as a string and parameters as a dict).
#     Subclass of Declarable can define its short name via a static variable 'short'.
#
#     For example let's define 3 classes. Note that parameters 'graph' and 'param' in Crawler are not
#     sent to superclass thus ignored (undeclared).
#
#     >>> class Crawler(Declarable):
#     >>>     def __init__(self, graph, a_list, b_dict, pred1, pred2, param, **kwargs):
#     >>>         self.graph = graph  # undeclared parameter
#     >>>         self.param = param  # undeclared parameter
#     >>>         super(Crawler, self).__init__(a_list=a_list, b_dict=b_dict, pred1=pred1, pred2=pred2, **kwargs)
#     >>>
#     >>> class Predictor(Declarable):
#     >>>     def __init__(self, x, param, **kwargs):
#     >>>         self.param = param  # undeclared parameter
#     >>>         super(Predictor, self).__init__(x = x, **kwargs)
#     >>>
#     >>> class GNN(Declarable): pass
#
#     Then create an object with two Declarable parameters, where the first one has its own Declarable
#     parameter. Print its declaration and filename.
#
#     >>> g = GraphCollections.get('example')
#     >>> c = Crawler(graph=g,
#     >>>             a_list=[1, 'a'],
#     >>>             b_dict={'b': 2},
#     >>>             pred1=Predictor(x=100, param=11, gnn=GNN(conv="SAGE")),
#     >>>             pred2=Predictor(x=200, param=22),
#     >>>             param="this is undeclared")
#     >>> print(c.declaration)
#     (<class '__main__.Crawler'>, {'a_list': [1, 'a'], 'b_dict': {'b': 2}, 'pred1': (<class '__main__.Predictor'>, {'x': 100, 'gnn': (<class '__main__.GNN'>, {'conv': 'SAGE'})}), 'pred2': (<class '__main__.Predictor'>, {'x': 200})})
#     >>> f = declaration_to_filename(c.declaration)
#     >>> print(f)
#     "Crawler(a_list=[1, 'a'];b_dict={'b': 2};pred1=Predictor@;pred2=Predictor@)/Predictor(gnn=GNN@;x=100)/GNN(conv=SAGE)/Predictor(x=200)"
#
#     In order to construct the same object as c, we should pass undeclared arguments:
#     >>> c_ = Declarable.from_declaration(
#     >>> filename_to_declaration(f),
#     >>> [(Crawler, {'graph': g, 'param': "redeclared"}),  # for Crawler
#     >>>  (Predictor, {'param': 11}),   # for 'pred1' parameter
#     >>>  (Predictor, {'param': 22})])  # for 'pred2' parameter
#     Note that if several parameters have same type, the order in the list should corresponds to the
#     order of parameters in constructor.
#
#     Declaration of the object contains recursively all declarations of each Declarable parameter
#     embedded.
#
#     Filename is built from a complex declaration using depth-first search. Filenames of embedded
#     declarations are separated with '/', i.e. corresponds to subfolders. '@' in a parameter name
#     marks an ambedded declaration.
#
#     NOTE: a problem can occur if one filename is longer than allowed by OS (usually 255 symbols).
#     """
#     @staticmethod
#     def is_declaration(obj):
#         """ Check if object is Declaration, i.e. tuple (class, Declaration).
#         """
#         return isinstance(obj, tuple) and len(obj) == 2 and \
#                isinstance(obj[0], type) and issubclass(obj[0], Declarable)
#
#     @staticmethod
#     def extract_declaration(obj):
#         """ Parse complex object, return its structure as is replacing Declarables with their
#         declarations.
#         Object can be arbitrary combination of allowed types.
#         """
#         allowed = (Declarable, type, Number, str, tuple, list, set, dict)
#         _kwargs = {}
#
#         if isinstance(obj, str):
#             if not re.match(r'[.\w]+', obj):
#                 raise DeclarableError("str arguments must be alpha-numeric with '.' and '_'")
#             return obj
#
#         if isinstance(obj, (Number, type)):
#             return obj
#         if isinstance(obj, Declarable):
#             return obj.declaration
#         if isinstance(obj, list):
#             return list(Declarable.extract_declaration(o) for o in obj)
#         if isinstance(obj, tuple):
#             return tuple(Declarable.extract_declaration(o) for o in obj)
#         if isinstance(obj, set):
#             return set(Declarable.extract_declaration(o) for o in obj)
#         if isinstance(obj, dict):
#             return {Declarable.extract_declaration(ok): Declarable.extract_declaration(ok)
#                     for ok, ov in obj.items()}
#
#         raise DeclarableError("Argument type %s is not allowed. Must be one of %s" % (
#             type(obj), [a.__name__ for a in allowed]))
#
#     def __init__(self, **kwargs):
#         """
#         List only those parameters you want to constitute filename.
#         If you need to ignore parameters like 'name', 'observed_set', etc. in file naming - don't send them.
#
#         Args:
#             **kwargs: parameters that will be used in file naming
#         """
#
#         # Recursively insert all inner declarations
#         _kwargs = {}
#         for key, val in kwargs.items():
#             # self._check_allowed(val)
#             _kwargs[key] = Declarable.extract_declaration(val)
#         self._declaration = type(self), _kwargs
#
#     @staticmethod
#     def from_declaration(declaration, aux_declarations: list=None, **aux_kwargs):
#         """ Build a class object from its declaration and auxiliary undeclared parameters, e.g. graph.
#         The declaration can contain inner declarations which will be converted recursively.
#         For convenience, aux_kwargs are aux_declarations for outermost class.
#         NOTE: aux_declarations can contain any arguments including not Declarable.
#
#         :param declaration: pair (class, kwargs)
#         :param aux_declarations: list of declarations in DFS order.
#         :param aux_kwargs: auxiliary keyword arguments for the outermost declaration.
#         :return:
#         """
#         if aux_declarations is None:
#             aux_declarations = []
#
#         _class, _kwargs = declaration
#
#         # Prepend aux_kwargs to aux_declarations. Just for convenience.
#         if len(aux_kwargs) > 0:
#             return Declarable.from_declaration(declaration, [(_class, aux_kwargs)] + aux_declarations)
#
#         # Pop arguments for this class
#         ix = -1
#         for i, (c, _) in enumerate(aux_declarations):
#             if c == _class:
#                 ix = i
#                 break
#         self_kwargs = aux_declarations.pop(ix)[1] if ix >= 0 else {}
#
#         # Get arguments for inner classes
#         kwargs = {}
#         for key, value in _kwargs.items():
#             if Declarable.is_declaration(value):
#                 value = Declarable.from_declaration(value, aux_declarations)
#
#             elif isinstance(value, list):
#                 value = [Declarable.from_declaration(obj, aux_declarations)
#                          if Declarable.is_declaration(obj) else obj for obj in value]
#
#             elif isinstance(value, tuple):
#                 value = tuple(Declarable.from_declaration(obj, aux_declarations)
#                          if Declarable.is_declaration(obj) else obj for obj in value)
#
#             elif isinstance(value, dict):
#                 final = {}
#                 for k, v in value.items():
#                     if Declarable.is_declaration(k):
#                         k = Declarable.from_declaration(k, aux_declarations)
#                     if Declarable.is_declaration(v):
#                         v = Declarable.from_declaration(v, aux_declarations)
#                     final[k] = v
#                 value = final
#
#             kwargs[key] = value
#
#         return _class(**self_kwargs, **kwargs)
#
#     @property
#     def declaration(self):
#         """ Get the declaration of this instance. """
#         return self._declaration
#
#
# def all_subclasses(cls):
#     """ Get all subclasses of a class. """
#     return set(cls.__subclasses__()).union(
#         [s for c in cls.__subclasses__() for s in all_subclasses(c)])
#
#
# class CrawlerException(Exception):
#     pass
#
#
# class DeclarableError(CrawlerException):
#     """ Can't convert object declaration to filename.
#     Possible reasons:
#
#     * filename too long (>255 symbols)
#     * object is not convertible
#     """
#     def __init__(self, error_msg=None):
#         super().__init__(self)
#         self.error_msg = error_msg if error_msg else "Couldn't build filename from object declaration."
#
#     def __str__(self):
#         return self.error_msg
#
#
# def declaration_to_filename(declaration) -> str:
#     """ Convert crawler string declaration into filename. Uniqueness is maintained
#     """
#     _class, kwargs = declaration
#     args = []
#     subfolders = []
#
#     def to_filename(obj):
#         if isinstance(obj, (str, Number, type(None))):
#             # obj = str(obj)
#             # if isinstance(val, (type, types.BuiltinFunctionType)):
#             #     # FIXME what if class name is ambiguous?
#             #     val = val.__name__
#             # elif hasattr(val, '__dict__') or hasattr(val, '__slots__'):
#             #     # Will be represented as "ClassName()", parameters are ignored
#             #     val = type(val).__name__ + "()"
#             # else:  # Primitive type expected
#             #     # FIXME : What if it's itself very long string?
#             #     val = str(val)
#             #     val = "".join([c for c in val if not c.isspace()])
#             if len(str(obj)) >= 255:
#                 raise DeclarableError("Filename for value is too long: '%s'" % obj)
#             return obj
#
#         # If obj is Declarable or declaration of Declarable
#         if isinstance(obj, Declarable) or Declarable.is_declaration(obj):
#             # Append declaration as a subfolder
#             subfolders.append(declaration_to_filename(
#                 obj.declaration if isinstance(obj, Declarable) else obj))
#             # Put type in place of value
#             obj_class = type(obj) if isinstance(obj, Declarable) else obj[0]
#             return "%s@" % (obj_class.short if hasattr(obj_class, "short") else obj_class.__name__)
#
#         if isinstance(obj, list):
#             return str(list(to_filename(o) for o in obj))
#         if isinstance(obj, tuple):
#             return str(tuple(to_filename(o) for o in obj))
#         if isinstance(obj, set):
#             return str(set(to_filename(o) for o in obj))
#         if isinstance(obj, dict):
#             return str({to_filename(ok): to_filename(ok) for ok, ov in obj.items()})
#
#         raise DeclarableError(f"Unexpected type '{type(obj)}' within a declaration")
#
#     for key in sorted(kwargs.keys()):
#         val = kwargs[key]
#         args.append("%s=%s" % (key, to_filename(val)))
#
#     args = ";".join(args)
#     main = "%s(%s)" % (_class.short if hasattr(_class, "short") else _class.__name__, args)
#     res = os.path.sep.join([main] + subfolders)
#     return res
#
#
# # short class name -> class
# short_to_class = {}
#
#
# def filename_to_declaration(filename):
#     """ Convert filename into crawler declaration. Uniqueness is maintained.
#     """
#     if isinstance(filename, str):
#         return filename_to_declaration(filename.split(os.path.sep))
#     assert isinstance(filename, list)
#
#     def eval_(string: str, filename):
#         """
#         Evaluates argument value from string.
#         Uses built-in eval, then checks if result is iterable and contains filename.
#         """
#         try:
#             res = eval(string)
#             # Check if it is an iterable of Declarables
#             if isinstance(res, list):
#                 return [filename_to_declaration(filename)
#                         if (isinstance(obj, str) and obj.endswith('@')) else obj for obj in res]
#
#             if isinstance(res, tuple):
#                 return tuple(filename_to_declaration(filename)
#                         if (isinstance(obj, str) and obj.endswith('@')) else obj for obj in res)
#
#             if isinstance(res, dict):
#                 final = {}
#                 for key, value in res.items():
#                     if isinstance(key, str) and key.endswith('@'):
#                         key = filename_to_declaration(filename)
#                     if isinstance(value, str) and value.endswith('@'):
#                         value = filename_to_declaration(filename)
#                     final[key] = value
#                 return final
#
#             return res
#
#         except NameError:
#             # Couldn't eval string as class name - return as is
#             return string
#
#     if len(short_to_class) == 0:
#         # Build short names dict
#         for sb in set().union(all_subclasses(Declarable)):
#             if hasattr(sb, 'short'):
#                 name = sb.short
#                 assert name not in short_to_class
#                 short_to_class[name] = sb
#             else:
#                 short_to_class[sb.__name__] = sb
#
#     # Recursive unpack
#     _class_str, params = re.findall(r"([^\(\)]*)\((.*)\)", filename.pop(0))[0]
#     _class = short_to_class[_class_str]
#     kwargs = {}
#     if len(params) > 0:
#         for assignment in params.split(';'):
#             key, value = assignment.split('=', 1)
#             if value.endswith('@'):  # is Declarable
#                 kwargs[key] = filename_to_declaration(filename)
#                 # NOTE: recursively included subfolders will be popped
#             else:
#                 kwargs[key] = eval_(value, filename)
#     return _class, kwargs
#
#
# # ------ Test classes
#
# class Crawler(Declarable):
#     def __init__(self, graph, a_list, b_dict, pred1, pred2, param, **kwargs):
#         self.param = param  # undeclared param
#         super(Crawler, self).__init__(a_list=a_list, b_dict=b_dict, pred1=pred1, pred2=pred2, **kwargs)
#         self.pred1 = pred1
#         self.pred2 = pred2
#
# class Predictor(Declarable):
#     def __init__(self, x, param, **kwargs):
#         self.param = param  # undeclared param
#         super(Predictor, self).__init__(x = x, **kwargs)
#
# class GNN(Declarable): pass
#
#
# if __name__ == '__main__':
#     print("declarable")
#
#     from sklearn.ensemble import GradientBoostingClassifier
#
#     xgb = eval('GradientBoostingClassifier')()
#     print(xgb)
#
#     def atest(obj):
#         d = obj.declaration
#         print("obj.declaration:", d)
#         f = declaration_to_filename(d)
#         print("filename from declaration:", f)
#         obj_ = Crawler.from_declaration(d, graph=g)
#         print("declaration of restored from_declaration:", obj_.declaration)
#         d_ = filename_to_declaration(f)
#         print("declaration of restored from filename:", d_)
#         f_ = declaration_to_filename(d_)
#         print("then to filename:", f_)
#         assert f == f_
#
#     from graph_io import GraphCollections
#     g = GraphCollections.get('konect', 'dolphins')
#
#     gnn = GNN(conv="GCN")
#     # gnn = GNN(conv="SAGE")
#     # p1 = Predictor(name="pred1", param="param", gnn=gnn)
#     # p2 = Predictor(name="pred2", param="var", param2=200)
#     # c = Crawler(graph="graph", crawled_set={1,2}, p1=p1, p2=p2, a=1, b=2)
#
#     cd = (Crawler, {'sub': 111})
#     # m = Crawler(graph="graph", crawler_decl=cd, feats=['a', 'BB'])
#     # test(p)
#     # test(c)
#     # test(m)
#
#     # c = Crawler(graph=g,
#     #             predictor1=Predictor(gnn=GNN(conv="SAGE")),
#     #             predictor2=Predictor(param=100),
#     #             a=1,
#     #             b=2)
#     # print(c.declaration)
#     # print(declaration_to_filename(c.declaration))
#
#     import torch
#     # a = torch.relu
#     # print(a.__name__)
#     # print(a.__class__)
#     # print(type(a))
#     # t = Crawler(graph=g, astr="abc", alist=[1, 'a'], aset={2, 'b'}, adict={'a': 1, 'b': "BB"})
#     # f = declaration_to_filename(t.declaration)
#     # res = Crawler.from_declaration(filename_to_declaration(f), graph=g)
#     # test(t)
#
#     c = Crawler(graph=g,
#                 a_list=[1, 'a'],
#                 b_dict={'b': 2},
#                 pred1=Predictor(x=10, param=1, gnn=GNN(conv="SAGE")),
#                 pred2=[Predictor(x=100, param=21), Predictor(x=200, param=22)],
#                 # pred2=Predictor(x=200, param=22),
#                 param="this is undeclared")
#     # import inspect
#     #
#     # signature = inspect.signature(Crawler.__init__).parameters
#     # for name, parameter in signature.items():
#     #     print(name, parameter.default, parameter.annotation, parameter.kind)
#
#     print(c.declaration)
#
#     f = declaration_to_filename(c.declaration)
#     print(f)
#     d = filename_to_declaration(f)
#     print(d)
#     assert declaration_to_filename(d) == f
#     c_ = Declarable.from_declaration(
#         d,
#         [(Crawler, {'graph': g, 'param': "redeclared"}),
#          (Predictor, {'param': 1}),
#          (Predictor, {'param': 21}),
#          (Predictor, {'param': 22})]
#     )
#     print(c_)
#     assert c.param != c_.param
#     assert c.pred1.param == c_.pred1.param
#     assert c.pred2[0].param == c_.pred2[0].param
#     f_ = declaration_to_filename(c_.declaration)
#     print(f_)
#     assert f == f_
