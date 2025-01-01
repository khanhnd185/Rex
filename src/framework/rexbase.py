
class RexBase:
  def __init__(self
               , standalone=False
               , arguments = {}
               , object_name="_"
               , class_name="__class__"
               ):
    self.arguments   = arguments
    self.class_name  = class_name
    self.object_name = object_name
    self.prefix      = "" if standalone else "self."

  def __str__(self):
    return self.class_name

  def gen_call(self, returns, args):
    r = ""
    if type(returns) == list:
      for ret in returns:
        r += f"{ret},"
    else:
      r = returns

    args = ",".join(args)
    return f"{r} = {self.prefix}{self.object_name}({args})"

  def gen_init(self):
    args = ",".join(f"{k}={str(v)}" for k,v in self.arguments.items())
    return f"{self.prefix}{self.object_name} = {self.class_name}({args})"
  
  def set_object_name(self, name):
    self.object_name = name

  def update_args(self, args):
    for k in self.arguments.keys():
      if k in args.keys():
        self.arguments[k] = str(args[k])


class RexDictBase(RexBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def gen_call(self, returns, args):
    r = ""
    if type(returns) == list:
      for ret in returns:
        r += f"{ret},"
    else:
      r = returns

    args = ",".join(args)
    return f"{r} = {self.prefix}d['{self.object_name}']({args})"

  def gen_init(self):
    args = ",".join(f"{k}={str(v)}" for k,v in self.arguments.items())
    return f"{self.prefix}d['{self.object_name}'] = {self.class_name}({args})"
