import { Injectable } from '@angular/core';
import { Location } from '@angular/common';
import { AppService } from './app.service';
import { AppSwal } from './app.swal';
import { Router } from '@angular/router';
import { AppConsts } from 'src/app/services/app.consts';
@Injectable({
    providedIn: 'root'
})
export class AppControls {

    constructor(
        public location: Location,
        private appService: AppService,
        private appSwal : AppSwal,
        private router: Router,
        public appConsts: AppConsts
    ) {

     }

    async getControls(roleID, pageUrl) {
        //let pageUrl = this.location.path();
        if (!pageUrl) {
            //pageUrl = '/home';
            // return null;

            //this.appSwal.showError('pageUrl không được phép rỗng');
            return;
        }
        // const idx = pageUrl.indexOf('?');
        // if (idx >= 0) {
        //     pageUrl = pageUrl.substring(1, idx);
        // } else {
        //     pageUrl = pageUrl.substring(1);
        // }
        let controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
        

        switch (pageUrl) {

            case 'portal/question/receive-questions': {
                controlDefault = ['SendQuestion'];
                break;
            }
            case 'portal/receive-document': {
                controlDefault = ['Lookup','Pay','Continue','Save','Delete'];
                break;
            }
            
            case 'cv/pro5-register/payment': {
                controlDefault = [];
                break;
            }

            case 'cv/pro5-register/unprocessed-document': {
                controlDefault = ['Details', 'Edit', 'Delete', 'UpdateProgress', 'History',
                                  'Deny','Approve', 'Confirm','Continue','Save',];
                break;
            }
            case 'cv/pro5-register': {
                controlDefault = ['AddNew'];
                break;
            }
            case 'cv/pro5-group': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Approve', 'UnApprove'];
                break;
            }
            case 'cv/pro5': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'ad/user-role': {
                controlDefault = ['Save'];
                break;
            }
            case 'ad/user': {
                controlDefault.push('AddCommon');
                break;
            }
            case 'profile/manage-personal': {
                controlDefault = ['AddNew', 'Edit', 'Delete'];
                break;
            }
            case 'profile/personal': {
                controlDefault = ['Save', 'Delete'];
                break;
            }

            case 'post/manage-post': {
                controlDefault = ['AddNew', 'Edit', 'Delete','Review' , 'Approve' , 'UnApprove'];
                break;
            }
            case 'post/post': {
                controlDefault = ['Save','Delete','Review'];
                break;
            }
            case 'cv/appointment-document': {
                controlDefault = ['UpdateProgress','ReleaseDocument','History'];
                break;
            }
            case 'report/report-unit': {
                controlDefault = ['ReportUnitSend' ,'ReportUnitReceive','ReportStatus'];
                break;
            }

        
           
            case 'meeting/meeting-public':
            case 'meeting/meeting': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Approve', 'UnApprove', 'ManageMember', 'ManageDocument', 'ManageQuestionBQ', 'ViewFile', 'ManageQuestionBC', 'ManageProcess', 'Insert','SearchByUnit', 'SearchByCurWeek', 'Move', 'CopyToVTTP', 'Handle'];
                break;
            }
            case 'ad/config': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }case 'profile/organization': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'meeting/document': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'meeting/process': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'm/archive': {
                controlDefault = ['AddFolder', 'AddFile', 'Handle', 'Share', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'meeting/question': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'meeting/member': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save'];
                break;
            }
            case 'm/common': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save']
                break;
            }
            case 'office/of-office': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Forward', 'AddCommon', 'End', 'Handle']
                break;
            }    
            case 'req/comment-list': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Handle', 'Info']
                break;
            }        
            case 'req/comment': {
                controlDefault = ['AddNew','End', 'Transfer', 'Reply', 'Next', 'Assign','EditEndDate','EditFeedbackDate']
                break;
            }
            case 'profile/personal-unit': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save']
                break;
            }
            case 'news-feed/news-list': {
                controlDefault = ['Edit', 'Delete']
                break;
            }
            case 'news-feed/news-info': {
                controlDefault = ['Image', 'Emoji','Video', 'Save']
                break;
            }
            case 'news-feed/news': {
                controlDefault = ['Image', 'Emoji','Video', 'Save']
                break;
            }
            case 'news-feed/news-history': {
                controlDefault = []
                break;
            }
            case 'rp/rp-dynamic': {
                controlDefault = ['Save', 'Send']
                break;
            }            

            case 'report/report-row': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save']
                break;
            }

            case 'report/rp-report-net': {
                controlDefault = []
                break;
            }
            case 'm/unit-contact': {
                controlDefault = ['AddNew', 'Handle', 'Edit', 'Delete', 'Save'];
                break;
            }

            case 'report/rp-template': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Setting']
                break;
            }
            
            case 'data/data-menu': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save', 'Setting']
                break;
            }
            
            case 'cv/processed-document': {
                controlDefault = ['Save', 'Progress','History','UnApprove']
                break;
            }

            case 'm/manage-image': {
                controlDefault = ['AddNew', 'Edit', 'Delete', 'Save']
                break;
            }

            default: {
                break;
            }
        }

        const dataRequest = {
            PageUrl: pageUrl,
            RoleID: roleID,
            Controls: controlDefault
        };
        const result = await this.appService.doPOST('api/Control/Check', dataRequest);
        const control = {};
        if (result && result.Status == 1 && result.Data && result.Data.length > 0) {
            result.Data.forEach(item => {
                control[item.ID] = item.ActiveFlg;
            });
        }

        return control;
    }

    async getPageName() {
        let pageUrl = this.location.path();
        if (!pageUrl) {
            return null;
        }

        const idx = pageUrl.indexOf('?');
        if (idx >= 0) {
            pageUrl = pageUrl.substring(0, idx);
        }

        pageUrl = pageUrl.substring(1);

        const dataRequest = {
            id: pageUrl,
            langID : localStorage.getItem('currentLanguage') ? localStorage.getItem('currentLanguage') : this.appConsts.defaultLangID
        };
        const result = await this.appService.doGET('api/Page/GetName', dataRequest);
        if (result && result.Status === 1) {
            return result.Data;
        }
        return '';
    }

    async getPageNamePortal() {
        let pageUrl = this.location.path();
        if (!pageUrl) {
            return null;
        }

        const idx = pageUrl.indexOf('?');
        if (idx >= 0) {
            pageUrl = pageUrl.substring(0, idx);
        }

        pageUrl = pageUrl.substring(1);

        const dataRequest = {
            id: pageUrl,
            langID : localStorage.getItem('portal') ? localStorage.getItem('portal') : this.appConsts.defaultLangID
        };
        const result = await this.appService.doGET('api/Page/GetName', dataRequest);
        if (result && result.Status === 1) {
            return result.Data;
        }
        return '';
    }
}