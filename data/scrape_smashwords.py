# Scrapes the BookCorpus dataset from smashwords.com
from bs4 import BeautifulSoup
from ebooklib import epub
import html2text
import aiohttp
from io import BytesIO
import sys, os, time, datetime, json, tempfile, zipfile, xml
import xml.parsers.expat
import asyncio
from urllib.parse import unquote

MAX_RETRY = 3

class ContainerParser():
    def __init__(self, xmlcontent=None):
        self.rootfile = ""
        self.xml = xmlcontent

    def startElement(self, name, attributes):
        if name == "rootfile":
            self.buffer = ""
            self.rootfile = attributes["full-path"]

    def parseContainer(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.Parse(self.xml, 1)
        return self.rootfile

class BookParser():
    def __init__(self, xmlcontent=None):
        self.xml = xmlcontent
        self.title = ""
        self.author = ""
        self.inTitle = 0
        self.inAuthor = 0
        self.ncx = ""

    def startElement(self, name, attributes):
        if name == "dc:title":
            self.buffer = ""
            self.inTitle = 1
        elif name == "dc:creator":
            self.buffer = ""
            self.inAuthor = 1
        elif name == "item":
            if attributes["id"] == "ncx" or attributes["id"] == "toc" or attributes["id"] == "ncxtoc":
                self.ncx = attributes["href"]

    def characters(self, data):
        if self.inTitle:
            self.buffer += data
        elif self.inAuthor:
            self.buffer += data

    def endElement(self, name):
        if name == "dc:title":
            self.inTitle = 0
            self.title = self.buffer
            self.buffer = ""
        elif name == "dc:creator":
            self.inAuthor = 0
            self.author = self.buffer
            self.buffer = ""

    def parseBook(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.EndElementHandler = self.endElement
        parser.CharacterDataHandler = self.characters
        parser.Parse(self.xml, 1)
        return self.title, self.author, self.ncx

class NavPoint():
    def __init__(self, id=None, playorder=None, level=0, content=None, text=None):
        self.id = id
        self.content = content
        self.playorder = playorder
        self.level = level
        self.text = text

class TocParser():
    def __init__(self, xmlcontent=None):
        self.xml = xmlcontent
        self.currentNP = None
        self.stack = []
        self.inText = 0
        self.toc = []

    def startElement(self, name, attributes):
        if name == "navPoint":
            level = len(self.stack)
            self.currentNP = NavPoint(
                attributes["id"], attributes["playOrder"], level)
            self.stack.append(self.currentNP)
            self.toc.append(self.currentNP)
        elif name == "content":
            self.currentNP.content = unquote(attributes["src"])
        elif name == "text":
            self.buffer = ""
            self.inText = 1

    def characters(self, data):
        if self.inText:
            self.buffer += data

    def endElement(self, name):
        if name == "navPoint":
            self.currentNP = self.stack.pop()
        elif name == "text":
            if self.inText and self.currentNP:
                self.currentNP.text = self.buffer
            self.inText = 0

    def parseToc(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.EndElementHandler = self.endElement
        parser.CharacterDataHandler = self.characters
        parser.Parse(self.xml, 1)
        return self.toc

def epub2txt(epub):
    with zipfile.ZipFile(epub, "r") as f:
        rootf = ContainerParser(
            f.read("META-INF/container.xml")).parseContainer()
        title, author, ncx = BookParser(f.read(rootf)).parseBook()
        ops = "/".join(rootf.split("/")[:-1])
        if ops != "":
            ops = ops+"/"
        toc = TocParser(f.read(ops + ncx)).parseToc()

        content = []
        for t in toc:
            html = f.read(ops + t.content.split("#")[0])
            text = html2text.html2text(html.decode("utf-8"))
            content.append("*" * (t.level+1) + " " +
                            t.text + "\n")
            content.append(t.text + "{{{%d\n" % (t.level+1))
            content.append(text + "\n")
    return ''.join(content)

async def extract_book(b_id, txt_url, epub_url, retry=0):
    async with aiohttp.ClientSession() as client:
        try:
            if txt_url:
                async with client.get(txt_url) as res:
                    with open('out/books/{}.txt'.format(b_id), 'wb') as f:
                        f.write(await res.read())
            else:
                async with client.get(epub_url) as res:
                    book = epub2txt(BytesIO(await res.read()))
                    with open('out/books/{}.epub.txt'.format(b_id), 'w') as f:
                        f.write(book)
            print('Downloaded {}'.format(b_id))
        except Exception as e:
            if retry == 0:
                print('Failed', b_id, e)
            else:
                print('Retry', b_id)
                return await extract_book(b_id, txt_url, epub_url, retry=retry-1)

async def extract_pages(book_links):
    async with aiohttp.ClientSession() as client:
        for b_link in book_links:
            b_url = b_link.get('href')
            b_id = b_url.split('/')[-1]
            print('Loading', b_id)

            async with client.get(b_url) as res:
                soup = BeautifulSoup(await res.read(), 'lxml')

                # get meta
                meta_infos = soup.find_all(class_="col-md-3")
                if not meta_infos:
                    sys.stderr.write('Failed: meta_info {}\n'.format(b_url))
                    continue
                meta_txts = [m.text for m in meta_infos if 'Language: English' in m.text]

                # check lang
                is_english = len(meta_txts) >= 1
                if not is_english:
                    continue
                
                # get txt if possible
                txt_url = None
                txt_links = soup.find_all(title="Archival; contains no formatting")
                if txt_links:
                    txt_url = txt_links[0].get('href')
                    if txt_url:
                        txt_url = 'https://www.smashwords.com' + txt_url

                # get epub
                epub_url = None
                epub_links = soup.find_all(title="Nook, Kobo, Sony Reader, and tablets")
                if epub_links:
                    epub_url = epub_links[0].get('href')
                    if epub_url:
                        epub_url = 'https://www.smashwords.com' + epub_url

                if not epub_url and not txt_url:
                    sys.stderr.write('Failed: epub and txt {}\n'.format(b_url))
                    continue

                asyncio.ensure_future(extract_book(b_id, txt_url, epub_url, retry=MAX_RETRY))

async def main():
    os.makedirs('out/books/', exist_ok=True)
    search_url_pt = 'https://www.smashwords.com/books/category/1/downloads/0/free/medium/{}'

    i = 0
    retry = MAX_RETRY

    while True:
        url = search_url_pt.format(i)

        async with aiohttp.ClientSession() as client:
            async with client.get(url) as res:
                soup = BeautifulSoup(await res.read(), 'lxml')
                book_links = soup.find_all(class_='library-title')

                if not book_links:
                    retry -= 1

                    if retry == 0:
                        print('Finished scraping.')
                        print(book_links)
                        break
                else:
                    asyncio.ensure_future(extract_pages(book_links))
                    i += 20
                    retry = MAX_RETRY
                    print('Page', i)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())