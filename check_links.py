import os
import re

def find_markdown_files(directory):
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def check_links(md_files):
    broken_links = []
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for file in md_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            links = link_pattern.findall(content)

            for text, link in links:
                if link.startswith('http://') or link.startswith('https://') or link.startswith('mailto:'):
                    continue
                if link.startswith('#'):
                    continue

                # Split anchor if present
                link_path = link.split('#')[0]
                if not link_path:
                    continue

                # Resolve relative path
                file_dir = os.path.dirname(file)
                target_path = os.path.join(file_dir, link_path)

                if not os.path.exists(target_path):
                    broken_links.append((file, link_path, target_path))

    return broken_links

if __name__ == '__main__':
    md_files = find_markdown_files('.')
    broken = check_links(md_files)
    if broken:
        print("Broken links found:")
        for file, link, target in broken:
            print(f"{file}: {link} -> {target}")
    else:
        print("No broken local links found.")
