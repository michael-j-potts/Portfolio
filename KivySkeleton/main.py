from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
import os
import threading

class DashBoard(Screen):
	pass

class LoginScreen(Screen):
	def try_login(self, login, pswrd):
		app = App.get_running_app()

		app.username = login
		app.password = pswrd
		self.manager.transition = SlideTransition(direction = 'up')
		self.manager.current = 'Dashboard'

		app.config.read(app.get_application_config())
		app.config.write()

	def reset(self):
		self.ids['username'].text = ''
		self.ids['password'].text = ''

class Metrics(Screen):
	pass

class Alerts(Screen):
	pass

class Resources(Screen):
	pass

class About(Screen):
	pass

class MyApp(App):
	username = StringProperty(None)
	password = StringProperty(None)

	def build(self):
		sm = ScreenManager()
		sm.add_widget(LoginScreen(name = 'Login'))
		sm.add_widget(DashBoard(name = 'Dashboard'))
		sm.add_widget(Metrics(name = 'Metrics'))
		sm.add_widget(Alerts(name = 'Alerts'))
		sm.add_widget(Resources(name = 'Resources'))
		sm.add_widget(About(name = 'About'))
		return sm

	def get_application_config(self):
		if(not self.username):
			return super(MyApp, self).get_application_config()
		
		conf_direct = self.user_data + '/' + self.username

		if(not os.path.exists(conf_direct)):
			os.makedirs(conf_direct)

		return super(MyApp, self).get_application_config(
			'%s/config.cfg' % (conf_direct)
		)


if __name__ == "__main__":
	MyApp().run()

