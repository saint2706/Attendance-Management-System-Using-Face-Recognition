import re

with open('frontend/src/pages/Dashboard.tsx', 'r') as f:
    content = f.read()

import_statement = "import { ActionCard } from '../components/ActionCard';\n"
if "ActionCard" not in content:
    content = content.replace("import { Link } from 'react-router-dom';", "import { Link } from 'react-router-dom';\n" + import_statement)

replacement = """<ActionCard to="/employees/register" title="Register a new employee" icon={UserPlus} heading="Register Employee" description="Add a new employee to the system" />
                    <ActionCard to="/add-photos" title="Capture photos for face recognition" icon={Camera} heading="Add Photos" description="Capture photos for face recognition" />
                    <ActionCard to="/train" title="Update the recognition model" icon={Brain} heading="Train Model" description="Update the recognition model" />
                    <ActionCard to="/attendance" title="Access attendance reports" icon={ChartBar} heading="View Attendance" description="Access attendance reports" />
                    <ActionCard to="/session" title="Monitor live recognition" icon={Radio} heading="Attendance Session" description="Monitor live recognition" />"""

# Using regex to find the actions-grid block
pattern = re.compile(r'<div className="actions-grid">.*?</section>', re.DOTALL)
match = pattern.search(content)

if match:
    old_grid = match.group(0)
    new_grid = '<div className="actions-grid">\n                    ' + replacement + '\n                </div>\n            </section>'
    content = content.replace(old_grid, new_grid)

with open('frontend/src/pages/Dashboard.tsx', 'w') as f:
    f.write(content)
