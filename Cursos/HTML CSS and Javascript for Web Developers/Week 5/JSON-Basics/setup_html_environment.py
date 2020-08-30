import os

# Creates index.html file
try:
    index_file = open("index.html", "w")
    index_file.write('''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>index</title>
        <link rel="stylesheet" type="text/css" href="css/style.css">
        <script type="text/javascript" src="js/script.js"></script>
    </head>
    <body>

        <h1>Title here</h1>
        
    </body>
    </html>
    ''')
    index_file.close()
except Exception:
    pass

# Creates directory for scripts
try:
    path = os.path.join(os.getcwd(), "js")
    os.mkdir(path)
except Exception:
    pass

# Creates script.js in js directory
try:
    js_file = open(os.path.join(path, "script.js"), "w")
    js_file.close()
except Exception:
    pass

# Creates directory for css
try:
    path = os.path.join(os.getcwd(), "css")
    os.mkdir(path)
except Exception:
    pass

# Creates style.css in css directory
try:
    css_file = open(os.path.join(path, "style.css"), "w")
    css_file.close()
except Exception:
    pass







