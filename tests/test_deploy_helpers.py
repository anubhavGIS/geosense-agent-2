# test_deploy_helpers.py
# Purpose: offline tests of the deployment helpers of Phase 5 (deploy/cloud_setup.py): how
#          the cloud database URL becomes the service's variables, and which of the two
#          addresses of the Cloud Run service the scripts use. No gcloud command is run and
#          no value of .env is read, so the CI pipeline runs them (Phase 5, Exercise 4).
# Run with: python -m pytest tests/test_deploy_helpers.py -v     (from the project root)
import sys

import pytest

sys.path.append('deploy')
import cloud_setup                                                    # noqa: E402

NEW_URL = 'https://geosense-backend-917306622260.us-central1.run.app'
OLD_URL = 'https://geosense-backend-ioq5ijky3q-uc.a.run.app'


def test_database_url_becomes_the_service_variables():
    db = cloud_setup.cloud_db('postgresql://neondb_owner:fake%40pw@ep-x-y.c-9.us-east-2.'
                              'aws.neon.tech/neondb?sslmode=require&channel_binding=require')
    assert db == {'host': 'ep-x-y.c-9.us-east-2.aws.neon.tech', 'port': '5432',
                  'dbname': 'neondb', 'user': 'neondb_owner', 'password': 'fake@pw',
                  'sslmode': 'require'}


def test_ssl_is_required_when_the_url_does_not_say():
    db = cloud_setup.cloud_db('postgresql://u:fake@ep-z.aws.neon.tech:6543/geo')
    assert (db['port'], db['dbname'], db['sslmode']) == ('6543', 'geo', 'require')


def test_a_url_without_a_password_is_refused():
    with pytest.raises(SystemExit):
        cloud_setup.cloud_db('postgresql://neondb_owner@ep-x.aws.neon.tech/neondb')


def test_the_project_number_address_is_preferred():
    service = {'metadata': {'annotations': {'run.googleapis.com/urls':
                                            f'["{NEW_URL}", "{OLD_URL}"]'}},
               'status': {'url': OLD_URL}}
    assert cloud_setup.pick_url(service) == NEW_URL
    assert cloud_setup.pick_url({'status': {'url': OLD_URL}}) == OLD_URL   # older services
    assert cloud_setup.pick_url({}) is None
