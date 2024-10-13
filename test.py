from webui import webui
MyWindow = webui.window()
MyWindow.bind('MyID', my_function)
MyWindow.show("MyHTML")
webui.wait()