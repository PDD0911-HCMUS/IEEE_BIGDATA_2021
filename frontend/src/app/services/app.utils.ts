import { Injectable, OnInit } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { IntlService } from '@progress/kendo-angular-intl';

@Injectable({
    providedIn: 'root'
})
export class AppUtils implements OnInit {

    ngOnInit(): void {
    }

    constructor(
        private translate: TranslateService,
        public intl: IntlService
    ) {
    }
}
