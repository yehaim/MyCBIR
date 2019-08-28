import time
import uuid
from flask import request
import json
from entry.return_entry import ErrorResult

@app.route('/search', methods=['POST'])
def gpu_search():
    res = {}
    try:
        seart_time1 = time.time()
        request_id = str(uuid.uuid1())

        try:
            organ_id = str(request.form['organ_id'])
        except KeyError as err:
            result = ErrorResult('200', 'organ_id is none', request_id).to_string()
            return json.dumps(result)
        try:
            token = str(request.form['token'])
        except KeyError as err:
            ErrorResult('200', 'token is none', request_id).to_string()
            token = None
        try:
            cat_id = str(request.form['cat_id'])
        except KeyError as err:
            logging.info('cat_id is None')
            cat_id = int(0)


        try:
            img = request.form['img']
        except KeyError as err:
            logging.info('search img img is None')
            img = None

        if img is None:
            try:
                img_url = str(request.form['img_url'])
            except KeyError as err:
                result = ErrorResult('200', 'igm and img_url is none', request_id).to_string()
                return json.dumps(result)
            req = urllib.request.urlopen(img_url)
            data_byte = req.read()
            img_arr = bytearray(data_byte)
            img = np.asarray(img_arr, dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            img = DataIO.get_image_file_from_post_form_data(img)

        try:
            crop = request.form['crop']
        except KeyError as err:
            logging.info('crop is None')
            crop=False
        try:
            rn = request.form['rn']
        except KeyError as err:
            result = ErrorResult('200', 'rn is none', request_id).to_string()
            rn = 100

        if crop:
            crop_time1 = time.time()
            region = yolo.detect_image(img)
            logging.info('crop,region:'+str(region))
            img = img.crop(region)
            crop_time2 = time.time()
            logging.info('crop time :'+str(crop_time2-crop_time1))
        # get data
        img = model.get_pre_m(img)
        feat = model.get_feature_with_pre_m(MOD, MAT, img)
        feat = model.get_feature_apply_gpu(MAT, feat, id=ID)
        # convert numpy array to string
        feat_byte = feat.tostring()
        feat_str = feat_byte.decode('ISO-8859-1')
        data = {'request_id':request_id,'organ_id': organ_id, 'cat_id': cat_id, 'feat_str': feat_str,'rn':rn}

        logging.info('search request:'+str({'request_id':request_id,'organ_id': organ_id, 'cat_id': cat_id, 'feat_str': feat_str,'rn':rn}))

        # post data
        url = 'http://{0}:{1}/{2}'.format(INDEX_DOMAIN, service_port,'query')
        res = post_data_by_url(url, data)

        result = res

        img_info = {}
        img_info['cat_id'] = cat_id
        img_info['region'] = ''
        img_info['img_list'] = res['img_list']
        result.pop('img_list')
        result['img_info'] = img_info
    except Exception as err:
        traceback.print_exc()
        result = ErrorResult('200', 'search img fail', request_id).to_string()
    try:
        executor.submit(insert_record,organ_id, OPERATION_SEARCH, request_id, cat_id)
        # insert_record(organ_id, 1, request_id, cat_id)
    except Exception as err:
        traceback.print_exc()
    seart_time2 = time.time()
    logging.info('search time:'+str(seart_time2-seart_time1))
    logging.info('search return:'+json.dumps(result))
    return json.dumps(result)
