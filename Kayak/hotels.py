import scrapy
import os
import logging
from scrapy.crawler import CrawlerProcess


ville_destination='Colmar'

class Hotels(scrapy.Spider):
    name = 'hotels'
    allowed_domain = ['www.booking.com/index.fr.html/']
    start_urls = ['https://www.booking.com/index.fr.html/']

    # Parse function for login
    def parse(self, response):
        # FormRequest used to login
        return scrapy.FormRequest.from_response(
            response,
            formdata={'ss': ville_destination},
            callback=self.search
        )

    # Call of the search function (callback) used after login
    def search(self, response):
        
        hotels = response.css('.sr_item')

        for h in hotels:
            yield {
                #CSS tags to be retrieved from the Booking.com site with the inspector function of the browser
                'name': h.css('.sr-hotel__name::text').get(), 
                'url': "https://www.booking.com" + h.css('.hotel_name_link').attrib["href"],
                'coords': h.css('.sr_card_address_line a').attrib["data-coords"],
                'score': h.css('.bui-review-score__badge::text').get(),
                'description': h.css('.hotel_desc::text').get()
                
            }
            
     # NEXT button of the site to use to retrieve all the information(as seen in the course)
        try:
            next_page = response.css('a.paging-next').attrib["href"]
        except KeyError:
            logging.info('No next page. Terminating crawling process.')
        else:
            yield response.follow(next_page, callback=self.search)
            
    
filename = "Hotels_" + ville_destination.replace(" ", "-") + ".json"

if filename in os.listdir('src/'):
    os.remove('src/' + filename)

process = CrawlerProcess(settings= {
    'USER_AGENT': 'Chrome/84.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'LOG_LEVEL': logging.INFO,
    "FEEDS": {
        'src/' + filename: {"format": "json"},
    }
})


process.crawl(Hotels)
process.stop()
process.start()
