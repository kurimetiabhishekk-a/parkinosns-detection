import sys
import re

with open('templates/layout.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace title and CSS
old_head = """    <title>SB Admin 2 - Blank</title>

    <!-- Custom fonts for this template-->
    <link href="vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
    <link
        href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i"
        rel="stylesheet">

    <!-- Custom styles for this template-->
    <link href="css/sb-admin-2.min.css" rel="stylesheet">

</head>"""

new_head = """    <title>ParkiSense</title>

    <!-- Custom fonts for this template-->
    <link href="{{ url_for('static', filename='vendor/fontawesome-free/css/all.min.css') }}" rel="stylesheet" type="text/css">
    <link
        href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i"
        rel="stylesheet">

    <!-- Custom styles for this template-->
    <link href="{{ url_for('static', filename='css/sb-admin-2.min.css') }}" rel="stylesheet">
    <link href="{{ url_for('static', filename='css/modern-theme.css') }}" rel="stylesheet">

</head>"""

content = content.replace(old_head, new_head)

# Replace Javascript paths at bottom
old_js = """    <!-- Bootstrap core JavaScript-->
    <script src="vendor/jquery/jquery.min.js"></script>
    <script src="vendor/bootstrap/js/bootstrap.bundle.min.js"></script>

    <!-- Core plugin JavaScript-->
    <script src="vendor/jquery-easing/jquery.easing.min.js"></script>

    <!-- Custom scripts for all pages-->
    <script src="js/sb-admin-2.min.js"></script>"""

new_js = """    <!-- Bootstrap core JavaScript-->
    <script src="{{ url_for('static', filename='vendor/jquery/jquery.min.js') }}"></script>
    <script src="{{ url_for('static', filename='vendor/bootstrap/js/bootstrap.bundle.min.js') }}"></script>

    <!-- Core plugin JavaScript-->
    <script src="{{ url_for('static', filename='vendor/jquery-easing/jquery.easing.min.js') }}"></script>

    <!-- Custom scripts for all pages-->
    <script src="{{ url_for('static', filename='js/sb-admin-2.min.js') }}"></script>"""

content = content.replace(old_js, new_js)

# Topbar Replacement Regex
new_topbar = """<!-- Topbar Welcome -->
                    <div class="d-none d-sm-inline-block form-inline mr-auto ml-md-3 my-2 my-md-0 mw-100">
                        <h5 class="mb-0 text-gray-800 font-weight-bold" style="letter-spacing: -0.5px;">Welcome to ParkiSense!</h5>
                    </div>

                    <!-- Topbar Navbar -->
                    <ul class="navbar-nav ml-auto">
                        <!-- Nav Item - User Information -->
                        <li class="nav-item dropdown no-arrow">
                            <a class="nav-link" href="#" id="userDropdown" role="button"
                                data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                <span class="mr-2 d-none d-lg-inline text-gray-600 font-weight-bold">Dr. Admin</span>
                                <div class="icon-circle bg-primary text-white" style="width:35px;height:35px;display:flex;align-items:center;justify-content:center;border-radius:50%;font-size:1rem;">
                                    <i class="fas fa-user-md"></i>
                                </div>
                            </a>
                        </li>
                    </ul>"""

content = re.sub(r'<!-- Topbar Search -->.*?</ul>', new_topbar, content, flags=re.DOTALL)

# Content area
old_content = """<!-- Begin Page Content -->
                <div class="container-fluid">

                    <!-- Page Heading -->
                    <h1 class="h3 mb-4 text-gray-800">Blank Page</h1>

                </div>
                <!-- /.container-fluid -->"""

new_content = """<!-- Begin Page Content -->
                <div class="container-fluid">
                    {% block content %}{% endblock %}
                </div>
                <!-- /.container-fluid -->"""

content = content.replace(old_content, new_content)

with open('templates/layout.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('layout.html updated')
