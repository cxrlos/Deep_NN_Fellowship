import numpy as np
import cv2 as cv
import spatial


def convert_google_to_standard(json_data, mode='ALL_VERTICES'):
    ''' Read JSON file returned by Google Detect Text API
    This function loads OCR data into our dict data structure
    TODO Verify for possible errors on json structure, e.g., missing keys
    '''
    nodes = []
    if mode == 'ALL_VERTICES':
        for page in json_data['fullTextAnnotation']['pages']:
            for block in page['blocks']:
                for paragraph in block['paragraphs']:
                    phrase = ''
                    pts = []
                    words = []
                    chars = []
                    chars_pts = []
                    confidence = []
                    end_flag = False
                    for word in paragraph['words']:
                        text = ''
                        pts_in_word = []
                        for symbol in word['symbols']:
                            symbol_pts = []
                            doit = True
                            for vertex in symbol['boundingBox']['vertices']:
                                if 'x' in vertex and 'y' in vertex:
                                    x = vertex['x']
                                    y = vertex['y']
                                    symbol_pts.append((x, y))
                                else:
                                    doit = False
                            if doit:
                                text += symbol['text']
                                chars.append(symbol['text'])
                                pts_in_word.extend(symbol_pts)
                                if 'confidence' in symbol:
                                    confidence.append(symbol['confidence'])
                                else:
                                    confidence.append(1.0)

                            if 'property' in symbol and 'detectedBreak' in symbol['property']:
                                if symbol['property']['detectedBreak']['type'] == 'LINE_BREAK' or symbol['property']['detectedBreak']['type'] == 'EOL_SURE_SPACE':
                                    end_flag = True
                                elif symbol['property']['detectedBreak']['type'] == 'SPACE':
                                    text += ' '
                                    if doit:
                                        words.append(text)
                                        pts.extend(spatial.rect_to_corners_pts(cv.boundingRect(np.array(pts_in_word))))
                                        words.append(' ')
                                        shifted_symbol_pts = [(x[0]+1, x[1]) for x in symbol_pts]
                                        pts.extend(shifted_symbol_pts)
                                        chars.append(' ')
                                        chars_pts.extend(pts_in_word)
                                        chars_pts.extend(shifted_symbol_pts)
                                        confidence.append(1.0)
                                        pts_in_word = []
                                            
                        if pts_in_word:
                            words.append(text)
                            pts.extend(spatial.rect_to_corners_pts(cv.boundingRect(np.array(pts_in_word))))
                            chars_pts.extend(pts_in_word)

                        phrase += text
                        if end_flag:
                            # print(phrase, pts, words)
                            # words.append(text)
                            # pts.extend(rect_to_corners_pts(cv.boundingRect(np.array(pts_in_word))))
                            # chars_pts.extend(pts_in_word)

                            if len(pts) >= 4:
                                nodes.append((phrase, pts, words, confidence, chars, chars_pts))

                            phrase = ''
                            pts = []
                            words = []
                            chars = []
                            chars_pts = []
                            confidence = []
                            end_flag = False
        words = []
        for token in nodes:
            r = cv.boundingRect(np.array(token[1]))
            # TODO check if necessary img_width, else remove
            img_width = 1000
            word = {'text': token[0], 'boundingBox': r, 'alignment_h': r[0] / img_width,
                    'pts': token[1], 'words': token[2], 'confidence':token[3], 'chars':token[4],
                    'chars_pts':token[5]}
            words.append(word)
        return words
    else:
        return None


