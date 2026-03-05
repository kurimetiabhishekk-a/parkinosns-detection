import sys
import re

with open('templates/home.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add Hero Banner
hero_banner = """{% extends 'layout.html' %}
{% block title %}ParkiSense Home{% endblock %}
{% block content %}

<!-- Hero Banner -->
<div class="card bg-gradient-primary text-white shadow-lg mb-4 glass-card" style="border-radius: 20px; overflow: hidden; position: relative;">
    <div style="position: absolute; top: -50%; left: -10%; width: 50%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%); transform: rotate(30deg); pointer-events: none;"></div>
    <div class="card-body p-5">
        <h1 class="display-4 font-weight-bold animate-fade-in" style="letter-spacing: -1px;">Hello, Dr. Admin!</h1>
        <p class="lead animate-fade-in" style="animation-delay: 0.1s; font-size: 1.15rem; opacity: 0.9;">Welcome to the ParkiSense analysis dashboard. Let's make an impact today.</p>
        <div class="mt-4 animate-fade-in" style="animation-delay: 0.2s;">
            <a href="/record" class="btn btn-modern bg-white text-primary shadow-sm mr-3" style="color: #6366f1 !important;"><i class="fas fa-microphone mr-2"></i> Voice Analysis</a>
            <a href="/upload" class="btn btn-modern shadow-sm" style="background: rgba(255,255,255,0.1) !important; color: #fff !important; border: 1px solid rgba(255,255,255,0.2) !important;"><i class="fas fa-pencil-alt mr-2"></i> Drawing Analysis</a>
        </div>
    </div>
</div>
"""

# Replace the beginning of the file up to the first row (excluding the row itself)
content = re.sub(r'^{% extends [^}]*%}.*?(<div class="row">)', hero_banner + r'\n\1', content, flags=re.DOTALL | re.MULTILINE)

# 2. Update the 4 Stat Cards
old_cards_regex = r'<div class="row">.*?<!-- Content Row -->'

new_cards = """<div class="row">

    <!-- Total Tests Run Card -->
    <div class="col-xl-3 col-md-6 mb-4 animate-slide-in" style="animation-delay: 0.1s;">
        <div class="card border-left-primary shadow h-100 py-2 glass-card">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-primary text-uppercase mb-1">Total Tests Run</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">1,245</div>
                    </div>
                    <div class="col-auto">
                        <i class="fas fa-vial fa-2x text-primary" style="opacity: 0.7;"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Image Model Accuracy Card -->
    <div class="col-xl-3 col-md-6 mb-4 animate-slide-in" style="animation-delay: 0.2s;">
        <div class="card border-left-success shadow h-100 py-2 glass-card">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-success text-uppercase mb-1">Image Model Accuracy</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">92.4%</div>
                    </div>
                    <div class="col-auto">
                        <i class="fas fa-brain fa-2x text-success" style="opacity: 0.7;"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Voice Model Accuracy Card -->
    <div class="col-xl-3 col-md-6 mb-4 animate-slide-in" style="animation-delay: 0.3s;">
        <div class="card border-left-info shadow h-100 py-2 glass-card">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-info text-uppercase mb-1">Voice Model Accuracy</div>
                        <div class="row no-gutters align-items-center">
                            <div class="col-auto">
                                <div class="h5 mb-0 mr-3 font-weight-bold text-gray-800">88.5%</div>
                            </div>
                            <div class="col">
                                <div class="progress progress-sm mr-2">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: 88%" aria-valuenow="88" aria-valuemin="0" aria-valuemax="100"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-auto">
                        <i class="fas fa-wave-square fa-2x text-info" style="opacity: 0.7;"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Dataset Size Card -->
    <div class="col-xl-3 col-md-6 mb-4 animate-slide-in" style="animation-delay: 0.4s;">
        <div class="card border-left-warning shadow h-100 py-2 glass-card">
            <div class="card-body">
                <div class="row no-gutters align-items-center">
                    <div class="col mr-2">
                        <div class="text-xs font-weight-bold text-warning text-uppercase mb-1">Dataset Size (Samples)</div>
                        <div class="h5 mb-0 font-weight-bold text-gray-800">18,500+</div>
                    </div>
                    <div class="col-auto">
                        <i class="fas fa-database fa-2x text-warning" style="opacity: 0.7;"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Content Row -->"""

content = re.sub(old_cards_regex, new_cards, content, flags=re.DOTALL)

with open('templates/home.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('home.html updated')
