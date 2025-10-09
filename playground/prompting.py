from assist.promptable import Promptable

class MyClass(Promptable):
    def my_prompt_file(self):
        return self.prompt_file()

mc = MyClass()
mc.prompt_for("test_template.md.jinja",
              here="Here it is")
