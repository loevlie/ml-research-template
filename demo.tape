# Tape for the README demo GIF.
# Render: cd /Users/dennisloevlie/PhD_Project/ml-project-template && vhs demo.tape

Output demo.gif

Set FontSize 14
Set Width 1200
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 20

# Start in an empty scratch dir
Hide
Type "cd $(mktemp -d) && clear"
Enter
Sleep 500ms
Show

Type "copier copy --trust --skip-tasks gh:loevlie/ml-research-template brain-scan-project"
Sleep 600ms
Enter
Sleep 3500ms

# project_name
Type "Brain Scan Classifier"
Sleep 300ms
Enter
Sleep 600ms

# package_name (accept derived default)
Enter
Sleep 500ms

# project_description
Type "Classifies brain MRI scans into diagnostic categories."
Sleep 300ms
Enter
Sleep 500ms

# author_name
Type "Dennis Loevlie"
Enter
Sleep 400ms

# author_email
Type "dennis@example.com"
Enter
Sleep 400ms

# python_version (default 3.11)
Enter
Sleep 500ms

# cuda_version (default cu124)
Enter
Sleep 500ms

# logger (default wandb)
Enter
Sleep 500ms

# include_example (default true)
Enter
Sleep 500ms

# include_dennys_rules (default false)
Enter
Sleep 2s

Type "cd brain-scan-project && ls"
Sleep 300ms
Enter
Sleep 2s

Type "ls src/brain_scan_classifier/"
Sleep 300ms
Enter
Sleep 2500ms
