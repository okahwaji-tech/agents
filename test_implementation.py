#!/usr/bin/env python3
"""
Test script for MkDocs implementation
Validates the redesigned documentation site before creating a PR
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any


class MkDocsTestSuite:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.docs_dir = self.root_dir / "docs"
        self.site_dir = self.root_dir / "site"
        self.results = []
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
    
    def test_file_structure(self):
        """Test that required files exist"""
        required_files = [
            "mkdocs.yml",
            "pyproject.toml",
            "docs/index.md",
            "docs/quick-start.md",
            "docs/roadmap.md",
            "docs/tags.md",
            "docs/stylesheets/extra.css",
            "docs/javascripts/mathjax.js",
            "docs/javascripts/progress-tracker.js",
            "docs/javascripts/study-components.js",
            "docs/overrides/partials/progress-tracker.html",
            ".github/workflows/deploy-docs.yml",
            ".lighthouserc.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.root_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log_test(
                "File Structure", 
                False, 
                f"Missing files: {', '.join(missing_files)}"
            )
        else:
            self.log_test("File Structure", True, "All required files present")
    
    def test_mkdocs_config(self):
        """Test MkDocs configuration validity"""
        try:
            # Read the file and check for basic structure
            with open(self.root_dir / "mkdocs.yml", 'r') as f:
                content = f.read()

            # Check for required sections in the content
            required_sections = ['site_name:', 'theme:', 'plugins:', 'nav:']
            missing_sections = [s for s in required_sections if s not in content]

            if missing_sections:
                self.log_test(
                    "MkDocs Config",
                    False,
                    f"Missing sections: {', '.join(missing_sections)}"
                )
            else:
                self.log_test("MkDocs Config", True, "Configuration structure is valid")

        except Exception as e:
            self.log_test("MkDocs Config", False, f"Error reading config: {e}")
    
    def test_build_process(self):
        """Test that MkDocs can build the site"""
        try:
            result = subprocess.run(
                ["mkdocs", "build", "--clean"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log_test("Build Process", True, "Site builds successfully")
            else:
                self.log_test(
                    "Build Process", 
                    False, 
                    f"Build failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            self.log_test("Build Process", False, "Build timed out")
        except Exception as e:
            self.log_test("Build Process", False, f"Build error: {e}")
    
    def test_javascript_syntax(self):
        """Test JavaScript files for syntax errors"""
        js_files = [
            "docs/javascripts/mathjax.js",
            "docs/javascripts/progress-tracker.js", 
            "docs/javascripts/study-components.js"
        ]
        
        for js_file in js_files:
            file_path = self.root_dir / js_file
            if file_path.exists():
                try:
                    # Basic syntax check - look for common issues
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for balanced braces
                    open_braces = content.count('{')
                    close_braces = content.count('}')
                    
                    if open_braces != close_braces:
                        self.log_test(
                            f"JS Syntax ({js_file})", 
                            False, 
                            f"Unbalanced braces: {open_braces} open, {close_braces} close"
                        )
                    else:
                        self.log_test(f"JS Syntax ({js_file})", True)
                        
                except Exception as e:
                    self.log_test(f"JS Syntax ({js_file})", False, f"Error: {e}")
            else:
                self.log_test(f"JS Syntax ({js_file})", False, "File not found")
    
    def test_css_syntax(self):
        """Test CSS files for basic syntax"""
        css_file = self.root_dir / "docs/stylesheets/extra.css"
        
        if css_file.exists():
            try:
                with open(css_file, 'r') as f:
                    content = f.read()
                
                # Basic syntax checks
                open_braces = content.count('{')
                close_braces = content.count('}')
                
                if open_braces != close_braces:
                    self.log_test(
                        "CSS Syntax", 
                        False, 
                        f"Unbalanced braces: {open_braces} open, {close_braces} close"
                    )
                else:
                    self.log_test("CSS Syntax", True)
                    
            except Exception as e:
                self.log_test("CSS Syntax", False, f"Error: {e}")
        else:
            self.log_test("CSS Syntax", False, "CSS file not found")
    
    def test_html_templates(self):
        """Test HTML template syntax"""
        template_file = self.root_dir / "docs/overrides/partials/progress-tracker.html"
        
        if template_file.exists():
            try:
                with open(template_file, 'r') as f:
                    content = f.read()
                
                # Basic HTML validation
                if '<div' in content and '</div>' in content:
                    self.log_test("HTML Templates", True)
                else:
                    self.log_test("HTML Templates", False, "Invalid HTML structure")
                    
            except Exception as e:
                self.log_test("HTML Templates", False, f"Error: {e}")
        else:
            self.log_test("HTML Templates", False, "Template file not found")
    
    def test_github_workflow(self):
        """Test GitHub Actions workflow syntax"""
        workflow_file = self.root_dir / ".github/workflows/deploy-docs.yml"

        if workflow_file.exists():
            try:
                # Read the file and check for basic structure
                with open(workflow_file, 'r') as f:
                    content = f.read()

                # Check required workflow sections in content
                required_sections = ['name:', 'on:', 'jobs:']
                missing_sections = [s for s in required_sections if s not in content]

                if missing_sections:
                    self.log_test(
                        "GitHub Workflow",
                        False,
                        f"Missing sections: {', '.join(missing_sections)}"
                    )
                else:
                    self.log_test("GitHub Workflow", True, "Workflow structure is valid")

            except Exception as e:
                self.log_test("GitHub Workflow", False, f"Error: {e}")
        else:
            self.log_test("GitHub Workflow", False, "Workflow file not found")
    
    def test_dependencies(self):
        """Test that all dependencies are properly specified"""
        try:
            import toml
            with open(self.root_dir / "pyproject.toml", 'r') as f:
                config = toml.load(f)
            
            # Check for docs dependencies
            if 'project' in config and 'optional-dependencies' in config['project']:
                docs_deps = config['project']['optional-dependencies'].get('docs', [])
                
                required_deps = ['mkdocs', 'mkdocs-material', 'mkdocs-macros-plugin']
                missing_deps = [dep for dep in required_deps 
                              if not any(dep in d for d in docs_deps)]
                
                if missing_deps:
                    self.log_test(
                        "Dependencies", 
                        False, 
                        f"Missing dependencies: {', '.join(missing_deps)}"
                    )
                else:
                    self.log_test("Dependencies", True)
            else:
                self.log_test("Dependencies", False, "No docs dependencies found")
                
        except Exception as e:
            self.log_test("Dependencies", False, f"Error: {e}")
    
    def test_content_structure(self):
        """Test content organization and structure"""
        # Check for key content files
        key_files = [
            "docs/study-guide/index.md",
            "docs/study-guide/week-1/index.md",
            "docs/progress/index.md",
            "docs/materials/math/index.md"
        ]
        
        existing_files = [f for f in key_files if (self.root_dir / f).exists()]
        
        if len(existing_files) >= len(key_files) * 0.5:  # At least 50% exist
            self.log_test("Content Structure", True, f"{len(existing_files)}/{len(key_files)} key files exist")
        else:
            self.log_test("Content Structure", False, f"Only {len(existing_files)}/{len(key_files)} key files exist")
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🧪 Running MkDocs Implementation Tests\n")
        
        self.test_file_structure()
        self.test_mkdocs_config()
        self.test_build_process()
        self.test_javascript_syntax()
        self.test_css_syntax()
        self.test_html_templates()
        self.test_github_workflow()
        self.test_dependencies()
        self.test_content_structure()
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result['passed']:
                    print(f"   - {result['test']}: {result['message']}")
        
        return failed_tests == 0


if __name__ == "__main__":
    tester = MkDocsTestSuite()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Ready for PR creation.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please fix issues before creating PR.")
        sys.exit(1)
