from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup, Tag
import trafilatura
from markdown_it import MarkdownIt

# Служит для извлечения содержательного текста из HTML разметки
# 1. С помощью BeautifulSoup из HTML удаляются ненужные тэги
# 2. С помощью trafilatura, содержимое конвертируется в markdown
# 3. С помощью markdown-it-py удаляются лишние разделы
# 4. Удаляются стили и лишние символы (типа \n)
def extract_text_from_html(html_content: str):
    soup = BeautifulSoup(html_content, "lxml")
    main = soup.find("main")
    header_title = ""
    if main:
        # Определяем заголовок H1, trafilatura почему-то пропускает его
        header_title = extract_header_title(main)
        # Удаляем сноски, цитаты изображения и другие ненужные тэги
        remove_unnecessary_tags(main)
        # Оставляем только секцию main
        html_content = f"<!DOCTYPE html><html lang=\"en\"><body>{str(main)}</body></html>"

    # Извлекаем содержательный текст в виде markdown
    md = trafilatura.extract(html_content, output_format="markdown", include_links=False, include_formatting=True)
    if md is None:
        md = ""

    # trafilatura почему-то пропускает H1, добавляем его вручную
    if header_title:
        md = f"# {header_title}\n\n{md}"

    # Удаляем ненужные разделы
    for title in ["Behind the scenes", "Appearances", "Sources", "Notes and references", "External links"]:
        md = remove_markdown_section(md, title)

    # Удаляем маркеры стиля
    md = remove_markdown_style_markers(md)

    # Удаляем лишние символы
    md = remove_unnecessary_symbols(md)

    return md

# Извлекает заголовок
def extract_header_title(main: Tag):
    h1 = main.find('h1', class_='page-header__title', id='firstHeading')
    if not h1: 
        return ""
    for span in h1.find_all("span"):
        span.decompose()
    return h1.get_text(strip=True)

# Удаляет лишние тэги с ненужной информацией
def remove_unnecessary_tags(main: Tag):
    for tag in main.find_all(["img", "picture", "source", "svg", "video", "audio", "iframe", "script", "style", "noscript"]):
        tag.decompose()
    for quote in main.find_all("div", class_="quote"):
        quote.decompose()
    for span in main.find_all("span", class_="mw-editsection"):
        span.decompose()
    for sup in main.find_all("sup", class_="reference"):
        sup.decompose()

# Даные уровней markdown документа
@dataclass
class MdHeadingInfo:
    level: int
    text: str
    start: int

# Удаляет раздел
def remove_markdown_section(md: str, title: str):
    lines = md.splitlines(keepends=True)
    tokens = MarkdownIt().parse(md)

    headings: list[MdHeadingInfo] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            headings.append(MdHeadingInfo(level=int(tok.tag[1]), text=tokens[i + 1].content.strip(), start=tok.map[0]))

    start = None
    end = None
    for cur in headings:
        if cur.text == title:
            start = cur.start
            end = len(lines)
            for other in headings:
                if other.start > start and other.level <= cur.level:
                    end = other.start
                    break
            break

    if start is None or end is None:
        return md
    
    return "".join(lines[:start] + lines[end:])

# Удаляет маркеры стиля
def remove_markdown_style_markers(md: str):
    md = re.sub(r"~~(.+?)~~", r"\1", md, flags=re.DOTALL)                               # ~~xxx~~
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md, flags=re.DOTALL)                           # **xxx**
    md = re.sub(r"__(.+?)__", r"\1", md, flags=re.DOTALL)                               # __xxx__
    md = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", md, flags=re.DOTALL)     # *xxx*
    md = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", md, flags=re.DOTALL)           # _xxx_
    md = re.sub(r"`(.+?)`", r"\1", md, flags=re.DOTALL)                                 # `xxx`
    return md

# Удаляет лишние символы
def remove_unnecessary_symbols(md: str):
    md = re.sub(r"(?:\\n)+", "", md)
    return md

# Сохраняет содержимое в файл
def save_text(file_path: str, text: str):
    print(f"{datetime.now().isoformat()} - Сохранение файла '{file_path}'")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

# Извлекает полезное содержимое из HTML документов и сохраняет его в файлы        
def extract_text_from_docs_in_folder(docs_fold_path):
    for file_path in Path(docs_fold_path).rglob("*.html"):
        if file_path.is_file():
            print(f"{datetime.now().isoformat()} - Обработка файла '{file_path}'")
            try:
                html_content = file_path.read_text(encoding="utf-8", errors="replace")
                text_content = extract_text_from_html(html_content)
                save_text(f"{file_path}.md", text_content)
            except Exception as err:
                print(f"При обработке файла произошла ошибка: {err}", file=sys.stderr)

if __name__ == "__main__":
    docs_fold_path = sys.argv[1]
    extract_text_from_docs_in_folder(docs_fold_path)