I have also been working with [kivy](www.kivy.org) to create python based android applications. While the diabetic foot monitor I am creating
is proprietary, I wanted to share the skeleton of my kivy app.

The dependencies required to build kivy include:
kivy
buildozer (and its dependencies listed below)
python3

To initialize the application, we follow the buildozer instructions [here](https://kivy.org/doc/stable/guide/packaging-android.html)

```
git clone https://github.com/kivy/buildozer.git
cd buildozer
sudo python setup.py install
```

Navigate to your project directory

```
buildozer init
```

This creates buildozer.spec which should be edited as necessary (aside from changing the name, default options are fine for now)

install [buildozers dependencies](https://buildozer.readthedocs.io/en/latest/installation.html#targeting-android)

plug in your android device

```
buildozer android debug deploy run
```

The APK file will be installed on your android device.
