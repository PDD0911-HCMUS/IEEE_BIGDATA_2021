import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";
import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { AppRoutingModule } from "./app-routing.module";
import { RouterModule, Routes } from "@angular/router";
import { DatePipe } from '@angular/common';
import {
  HttpClientModule,
  HTTP_INTERCEPTORS,
  HttpClient,
} from "@angular/common/http";
import { HttpClientInMemoryWebApiModule } from "angular-in-memory-web-api";
import {
  Headers,
  Http,
  RequestOptions,
  Response,
  HttpModule,
} from "@angular/http";
import { FormsModule } from "@angular/forms";
import {
  MatPaginatorModule,
  MatProgressBarModule,
  MatSnackBarModule,
  MatSortModule,
  MatTableModule,
  MatDatepickerModule,
} from "@angular/material";
import { TranslateModule, TranslateLoader } from "@ngx-translate/core";
import { TranslateHttpLoader } from "@ngx-translate/http-loader";
import { AngularFontAwesomeModule } from "angular-font-awesome";
import { BsDropdownModule } from "ngx-bootstrap/dropdown";
import { FocusModule } from "angular2-focus";
// import { TooltipModule } from 'ngx-bootstrap/tooltip';
import { ModalModule } from "ngx-bootstrap/modal";

import { AppRouter } from "./app.router";
import { TokenInterceptor } from "./services/token.interceptor";
import { AuthService } from "./services/auth.service";
import { AppLanguage } from "./services/app.language";

import { AppComponent } from "./app.component";

import { DropDownsModule } from "@progress/kendo-angular-dropdowns";
import {
  DateInputModule,
  DatePickerModule,
} from "@progress/kendo-angular-dateinputs";
import { InputsModule } from "@progress/kendo-angular-inputs";
import { ButtonsModule } from "@progress/kendo-angular-buttons";
import { GridModule, ExcelModule } from "@progress/kendo-angular-grid";
import { LayoutModule } from "@progress/kendo-angular-layout";
import { ToolBarModule } from "@progress/kendo-angular-toolbar";
import { TooltipModule } from "@progress/kendo-angular-tooltip";
import { ExcelExportModule } from "@progress/kendo-angular-excel-export";
import { IntlModule } from "@progress/kendo-angular-intl";
import { NotificationModule } from "@progress/kendo-angular-notification";
import { UploadModule } from "@progress/kendo-angular-upload";
import { DialogModule, DialogsModule } from "@progress/kendo-angular-dialog";
import { PopupModule } from "@progress/kendo-angular-popup";
import { TreeViewModule } from "@progress/kendo-angular-treeview";
import { TreeListModule } from "@progress/kendo-angular-treelist";
import { ScrollViewModule } from '@progress/kendo-angular-scrollview';
import { ChartsModule } from "@progress/kendo-angular-charts";
import { QRCodeModule } from "angular2-qrcode";
import { Subject, from, merge, Observable } from "rxjs";
import { switchMap, map, windowCount, scan, take, tap } from "rxjs/operators";
import {
  ChatModule,
  Message,
  User,
  Action,
  ExecuteActionEvent,
  SendMessageEvent,
} from "@progress/kendo-angular-conversational-ui";

//đoiỉ tên tháng
import { LOCALE_ID } from "@angular/core";
import { DateInputsModule } from "@progress/kendo-angular-dateinputs";
import {
  LocationStrategy,
  PathLocationStrategy,
  registerLocaleData,
} from "@angular/common";
import localeVn from "@angular/common/locales/vi";
registerLocaleData(localeVn);

import '@progress/kendo-angular-intl/locales/en/all';
import "hammerjs";
import { NgxExtendedPdfViewerModule } from "ngx-extended-pdf-viewer";
import { DragDropModule } from "@angular/cdk/drag-drop";
import { WebSocketService } from "src/app/websocket.service";
import { CKEditorModule } from '@ckeditor/ckeditor5-angular';
import { OrgChartModule } from "@mondal/org-chart";

import { InfiniteScrollModule } from 'ngx-infinite-scroll';
import { VideojsComponent } from './videojs/videojs.component';
import { ModelRunComponent } from './run/model-run/model-run.component';

@NgModule({
  declarations: [
    AppComponent,
    VideojsComponent,
    ModelRunComponent,
  ],
  imports: [
    BrowserModule,
    ButtonsModule,
    DropDownsModule,
    DateInputModule,
    DatePickerModule,
    InfiniteScrollModule,
    GridModule,
    ExcelModule,
    ExcelExportModule,
    LayoutModule,
    //SchedulerModule,
    ToolBarModule,
    TooltipModule,
    QRCodeModule,
    UploadModule,
    DialogModule,
    CKEditorModule,
    TreeListModule,
    DialogsModule,
    PopupModule,
    TreeViewModule,
    TreeListModule,
    ScrollViewModule,
    ChartsModule,
    MatDatepickerModule,
    IntlModule,
    InputsModule,
    NotificationModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    HttpClientModule,
    HttpClientInMemoryWebApiModule,
    HttpModule,
    FormsModule,
    MatPaginatorModule,
    MatProgressBarModule,
    MatSnackBarModule,
    MatSortModule,
    MatTableModule,
    AngularFontAwesomeModule,
    RouterModule.forRoot(AppRouter.routes, { useHash: true }),
    TranslateModule.forRoot({
      loader: {
        provide: TranslateLoader,
        useFactory: HttpLoaderFactory,
        deps: [HttpClient],
      },
    }),
    BsDropdownModule.forRoot(),
    FocusModule.forRoot(),
    ModalModule.forRoot(),
    ChatModule,

    BrowserAnimationsModule,
    IntlModule,
    DateInputsModule,
    DragDropModule,
    NgxExtendedPdfViewerModule,
    OrgChartModule,
  ],
  exports: [BsDropdownModule, TooltipModule, ModalModule],
  providers: [
    AuthService,
    AppLanguage,
    DatePipe,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: TokenInterceptor,
      multi: true,
    },
    // {provide: LocationStrategy, useClass: PathLocationStrategy},
    WebSocketService,
    { provide: LOCALE_ID, useValue: "en-EN" },
  ],
  bootstrap: [AppComponent],
})
export class AppModule {}
export function HttpLoaderFactory(http: HttpClient) {
  return new TranslateHttpLoader(http);
}
